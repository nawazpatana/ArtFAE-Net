"""Verify a locked protocol before any held-out target loading or prediction."""
from pathlib import Path
import hashlib
import json
from artfae import EMOTIONS


def sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda:file.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def validate_protocol(path: Path,checkpoint: Path,data_dir: Path) -> dict:
    """Validate immutable selection evidence and file identities; never parse test labels."""
    protocol=json.loads(path.read_text())
    if protocol.get('status')!='locked' or protocol.get('selection_split')!='val':
        raise ValueError('Held-out evaluation requires a locked validation-selected protocol')
    if protocol.get('emotions')!=list(EMOTIONS):raise ValueError('Protocol class order differs')
    if not protocol.get('selection_rationale') or not protocol.get('limitations'):
        raise ValueError('Protocol must document model selection and limitations')
    if sha256(checkpoint)!=protocol.get('checkpoint_sha256'):raise ValueError('Selected checkpoint changed')
    validation=protocol.get('validation_evidence',{})
    evidence=Path(validation.get('path',''))
    if not evidence.is_file() or sha256(evidence)!=validation.get('sha256'):raise ValueError('Validation selection evidence changed or missing')
    report=json.loads(evidence.read_text())
    if report.get('macro_f1') is None or report.get('classes')!=list(EMOTIONS):raise ValueError('Invalid validation evidence')
    if report.get('checkpoint_sha256')!=protocol['checkpoint_sha256']:raise ValueError('Validation evidence is not bound to the selected checkpoint')
    if set(protocol.get('manifest_sha256',{}))!={'train','val','test'}:raise ValueError('Protocol must identify all fixed manifests')
    for split in ('train','val','test'):
        if sha256(data_dir/f'{split}_artworks.csv')!=protocol['manifest_sha256'][split]:raise ValueError(f'{split} manifest changed')
    return protocol
