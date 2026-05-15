from utils import get_numclasses
from utils.registry import Registry
import models

ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = """
Registry for encoder
"""

__all__ = ['build_encoder']

def build_encoder(args):

    num_classes = get_numclasses(args)

    if args.verbose:
        print(ENCODER_REGISTRY)

    print(f"=> Creating model '{args.model.name}, pretrained={args.model.pretrained}'")
    
    encoder = ENCODER_REGISTRY.get(args.model.name)(args, num_classes, **args.model) if len(args.model.name) > 0 else None

    return encoder
