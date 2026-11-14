"""Conservative explanations grounded in measurable attributes."""
import numpy as np
from artfae import EMOTIONS
from artfae.features.aesthetic import NAMES,describe


def explain(probabilities: np.ndarray, attributes: np.ndarray, regions: list, image_size: tuple[int,int] | None=None) -> dict:
    probabilities=np.asarray(probabilities,dtype=float)
    if probabilities.shape != (9,) or not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities>1).any() or not np.isclose(probabilities.sum(),1): raise ValueError('Invalid probabilities')
    attributes=np.asarray(attributes,dtype=float)
    if attributes.shape!=(len(NAMES),) or not np.isfinite(attributes).all():raise ValueError('Invalid raw attribute measurements')
    measured=dict(zip(NAMES,map(float,attributes)));tolerance=1e-4
    for name,value in measured.items():
        minimum=-1 if name in ('hue_sin','hue_cos') else 0
        if value<minimum-tolerance or value>1+tolerance:raise ValueError(f'Attribute outside measured range: {name}')
    if measured['warm']+measured['cool']>1+tolerance or measured['dark']+measured['bright']>1+tolerance:raise ValueError('Contradictory pixel proportions')
    if measured['saturation_var']>.25+tolerance or measured['luminance_std']>.5+tolerance:raise ValueError('Impossible bounded-pixel dispersion')
    if measured['luminance']<.8*measured['bright']-tolerance or measured['luminance']>1-.8*measured['dark']+tolerance:raise ValueError('Brightness proportions contradict mean luminance')
    order=np.argsort(-probabilities,kind='stable');predicted=int(order[0]);label=EMOTIONS[predicted]
    if image_size is not None and (len(image_size)!=2 or min(image_size)<=0):raise ValueError('Invalid decoded image dimensions')
    seen=set();mass=0.;geometry_checked=bool(regions) and image_size is not None;class_checked=bool(regions)
    for region in regions:
        index=region.get('patch_index');weight=region.get('attention')
        if not isinstance(index,int) or index<0 or index in seen or weight is None or not np.isfinite(weight) or not 0<=weight<=1:raise ValueError('Invalid candidate attention region')
        seen.add(index);mass+=weight
        if 'class_index' in region:
            if region['class_index']!=predicted:raise ValueError('Candidate regions belong to another emotion')
        else:class_checked=False
        if 'box_xyxy' in region:
            box=np.asarray(region['box_xyxy'],dtype=float)
            if box.shape!=(4,) or not np.isfinite(box).all() or (box<0).any() or box[2]<=box[0] or box[3]<=box[1]:raise ValueError('Invalid candidate region geometry')
            if image_size is not None and (box[2]>image_size[0]+tolerance or box[3]>image_size[1]+tolerance):raise ValueError('Candidate region outside decoded artwork')
        else:geometry_checked=False
    if mass>1+tolerance:raise ValueError('Selected attention exceeds normalized map mass')
    text=f'The model predicts {label}. Measured image properties: {describe(attributes)}'
    if geometry_checked:
        box=max(regions,key=lambda region:region['attention'])['box_xyxy'];cx=(box[0]+box[2])/(2*image_size[0]);cy=(box[1]+box[3])/(2*image_size[1])
        horizontal='left' if cx<1/3 else 'right' if cx>2/3 else 'center'
        vertical='upper' if cy<1/3 else 'lower' if cy>2/3 else 'middle'
        text+=f' The highest-ranked candidate region is in the {vertical} {horizontal} of the image.'
    entropy=float(np.clip(-np.sum(np.where(probabilities>0,probabilities*np.log(np.maximum(probabilities,1e-30)),0))/np.log(len(EMOTIONS)),0,1))
    return {'predicted_emotion':label,'probability':float(probabilities[predicted]),'probabilities':dict(zip(EMOTIONS,map(float,probabilities))),'other_prominent_emotions':[EMOTIONS[i] for i in order[1:3]],'measured_attributes':measured,'regions':regions,'explanation':text,'consistency_checks':{'raw_attributes_valid':True,'attribute_text_matches_measurements':True,'regions_valid':True,'region_bounds_checked':geometry_checked,'region_class_checked':class_checked,'selected_attention_mass':float(mass)},'uncertainty':{'normalized_predictive_entropy':entropy,'top_two_probability_margin':float(probabilities[order[0]]-probabilities[order[1]]),'interpretation':'Describes the model distribution, not observer agreement or calibrated reliability.'},'limitations':'Emotion is subjective. Measurements do not establish emotional causation. Attention is candidate evidence requiring intervention tests.'}
