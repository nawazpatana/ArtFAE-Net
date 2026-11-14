"""Invert CLIP resize/center-crop coordinates onto the decoded original artwork."""
from dataclasses import dataclass,asdict
import numpy as np
from PIL import Image

@dataclass(frozen=True)
class CLIPGeometry:
    original_size: tuple[int,int]
    resized_size: tuple[int,int]
    crop_size: tuple[int,int]
    left: int
    top: int

    @classmethod
    def from_processor(cls,image: Image.Image,processor) -> 'CLIPGeometry':
        if not processor.do_resize or not processor.do_center_crop:raise ValueError('Expected CLIP resize and center crop')
        width,height=image.size;size=processor.size
        if 'shortest_edge' in size:
            short=int(size['shortest_edge'])
            resized=(short,int(short*height/width)) if width<=height else (int(short*width/height),short)
        else:resized=(int(size['width']),int(size['height']))
        crop=(int(processor.crop_size['width']),int(processor.crop_size['height']))
        if resized[0]<crop[0] or resized[1]<crop[1]:raise ValueError('Padded crops require a separate inverse geometry')
        if resized[0]*resized[1]>40_000_000:raise ValueError('Extreme aspect ratio exceeds geometry memory limit')
        return cls(image.size,resized,crop,(resized[0]-crop[0])//2,(resized[1]-crop[1])//2)

    def map_values(self,values: np.ndarray,nearest: bool=False) -> np.ndarray:
        """Map a crop/patch grid back; pixels outside the encoder crop have zero map value."""
        if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('Invalid spatial map')
        mode=Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
        crop=Image.fromarray(values.astype(np.float32)).resize(self.crop_size,mode)
        canvas=Image.new('F',self.resized_size,0);canvas.paste(crop,(self.left,self.top))
        return np.asarray(canvas.resize(self.original_size,mode)).copy()

    def map_mask(self,mask: np.ndarray) -> np.ndarray:
        return self.map_values(mask.astype(np.float32),nearest=True)>.5

    def visible_mask(self) -> np.ndarray:
        return self.map_mask(np.ones((self.crop_size[1],self.crop_size[0]),dtype=bool))

    def patch_box(self,index: int,shape: tuple[int,int]) -> list[float]:
        height,width=shape
        if not 0<=index<height*width:raise ValueError('Patch index outside grid')
        row,column=divmod(index,width);cw,ch=self.crop_size;rw,rh=self.resized_size;ow,oh=self.original_size
        return [(self.left+column*cw/width)*ow/rw,(self.top+row*ch/height)*oh/rh,(self.left+(column+1)*cw/width)*ow/rw,(self.top+(row+1)*ch/height)*oh/rh]

    def to_dict(self) -> dict:return asdict(self)
