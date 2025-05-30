import argparse
from utils.utils import *
from trainer import Trainer
from action_extractor.nn.config import *

'''
Temporary
'''
oscar = False
if oscar:
    dp = "/users/ysong135/scratch/datasets/train"
    vp = "/users/ysong135/scratch/datasets/val"
    b = 16
    rp = '/users/ysong135/Documents/action_extractor/results'
else:
    dp = '/home/yilong/Documents/ae_data/random_processing/iiwa16168'
    dp = '/home/yilong/Documents/ae_data/random_processing/lift_1000'
    dp = '/home/yilong/Documents/policy_data/lift/raw/1736557347_6730564/test'
    dp = '/home/yilong/Documents/policy_data/lift/raw/1736991916_9054875/test/'
    # dp = '/home/yilong/Documents/policy_data/lift/obs'
    # vp = '/home/yilong/Documents/ae_data/abs'
    # vp = '/home/yilong/Documents/ae_data/random_processing/obs_rel_color2'
    vp = '/home/yilong/Documents/policy_data/lift/validate'
    # vp = '/home/yilong/Documents/policy_data/lift/obs'
    b = 16
    rp = '/home/yilong/Documents/action_extractor/results'

'''
Temporary
'''

def train(args):

    model_name = f'''{args.note}'''

    # Instantiate model
    model = load_model(
        args.architecture,
        horizon=args.horizon,
        results_path=args.results_path,
        latent_dim=args.latent_dim,
        cameras=args.cameras,
        motion=args.motion,
        image_plus_motion=args.image_plus_motion,
        num_mlp_layers=args.num_mlp_layers,
        vit_patch_size=args.vit_patch_size,
        resnet_layers_num=args.resnet_layers_num,
        idm_model_name=args.idm_model_name,
        fdm_model_name=args.fdm_model_name,
        freeze_idm=args.freeze_idm,
        freeze_fdm=args.freeze_fdm,
        action_type=args.action_type,
        data_modality=args.data_modality,
        vMF_sample_method=args.vMF_sample_method,
        spatial_softmax=args.spatial_softmax,
        )

    # Instandiate datasets
    train_set, validation_set = load_datasets(
        args.architecture, 
        args.datasets_path, 
        args.valsets_path,
        train=True,
        validation=True,
        horizon=args.horizon,
        demo_percentage=args.demo_percentage,
        num_demo_train = args.num_demo_train,
        val_demo_percentage=args.val_demo_percentage,
        cameras=args.cameras,
        motion=args.motion,
        image_plus_motion=args.image_plus_motion,
        action_type=args.action_type,
        data_modality=args.data_modality,
        compute_stats=args.standardize_data,
        coordinate_system=args.coordinate_system
        )

    # Instantiate the trainer
    trainer = Trainer(
        model, 
        train_set, 
        validation_set, 
        results_path=args.results_path, 
        model_name=model_name, 
        optimizer_name=args.optimizer,
        batch_size=args.batch_size, 
        epochs=args.epoch,
        lr=args.learning_rate,
        loss=args.loss,
        vae=isinstance(model, ActionExtractionVariationalResNet) or isinstance(model, ActionExtractionHypersphericalResNet),
        num_gpus=args.num_gpus
    )

    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Train the model
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train action extraction model")

    parser.add_argument(
        '--architecture', '-a', 
        type=str, 
        default='direct_resnet_mlp', 
        choices=ARCHITECTURES,
        help='Model architecture to train'
    )
    parser.add_argument(
        '--datasets_path', '-dp', 
        type=str, 
        default=dp, 
        help='Path to the datasets'
    )
    parser.add_argument(
        '--valsets_path', '-vp',
        type=str,
        default=vp,
        help='Path to the validation sets'
    )
    parser.add_argument(
        '--data_modality', '-dm',
        type=str,
        default='rgb',
        choices=['rgb', 'rgbd', 'voxel', 'color_mask_depth', 'cropped_rgbd', 'cropped_rgbd+color_mask', 'cropped_rgbd+color_mask_depth'],
        help='Type of data to use for training'
    )
    parser.add_argument(
        '--results_path', '-rp', 
        type=str, 
        default=rp,
        help='Path to where the results should be stored'
    )
    parser.add_argument(
        '--latent_dim', '-ld', 
        type=int, 
        default=32,
        choices=VALID_LATENT_DIMS,
        help='Latent dimension (sqrt of size)'
    )
    parser.add_argument(
        '--epoch', '-e', 
        type=int, 
        default=1, 
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--batch_size', '-b', 
        type=int, 
        default=b, 
        help='Batch size'
    )
    parser.add_argument(
        '--motion', '-m', 
        action='store_true', 
        help='Train only with motion'
    )
    parser.add_argument(
        '--image_plus_motion', '-ipm', 
        action='store_true', 
        help='Add motion preprocess to training data'
    )
    parser.add_argument(
        '--horizon', '-hr', 
        type=int, 
        default=2, 
        help='Length of the video'
    )
    parser.add_argument(
        '--idm_model_name', '-idm',
        type=str, 
        default='', 
        help='Path to pretrained latent IDM model'
    )
    parser.add_argument(
        '--fdm_model_name', '-fdm',
        type=str,
        default='',
        help='Path to pretrained latent FDM model'
    )
    parser.add_argument(
        '--freeze_idm', '-fidm',
        action='store_true',
        help='Freeze IDM model for auxiliary training'
    )
    parser.add_argument(
        '--freeze_fdm', '-ffdm',
        action='store_true',
        help='Freeze FDM model for auxiliary training'
    )
    parser.add_argument(
        '--demo_percentage', '-dpc',
        type=float,
        default=None,
        help='Percentage of demos (spread evenly across each task) to use for training'
    )
    parser.add_argument(
        '--num_demo_train',
        type=int,
        default=5000,
    )
    parser.add_argument(
        '--val_demo_percentage', '-vdp',
        type=float,
        default=0.9,
        help='Percentage of demos (spread evenly across each task) to use for validating'
    )
    parser.add_argument(
        '--vit_patch_size', '-vps',
        type=int,
        default=16,
        help='Patch size to use for the ViT if architecture involves a ViT component'
    )
    parser.add_argument(
        '--resnet_layers_num', '-rln',
        type=int,
        default=18,
        choices=[0, 18, 34, 50, 101, 152, 200],
        help='Number of layers if direct_resnet_mlp architecture is chosen'
    )
    parser.add_argument(
        '--note',
        type=str,
        default='',
        help='Custom note added to the end of the model name'
    )
    parser.add_argument(
        '--cameras', '-c',
        type=str,
        default='frontview_image',
        help='Comma separated list of camera angles to be used for training'
    )
    parser.add_argument(
        '--embodiments', '-emb',
        type=str,
        default='',
        help='Comma separated list of embodiments to be used for training'
    )
    parser.add_argument(
        '--optimizer', '-optim',
        type=str,
        default='adam',
        choices=['adam', 'sgd', 'rmsprop', 'adagrad', 'adamw'],
        help='Optimizer to use for training'
    )
    parser.add_argument(
        '--learning_rate', '-lr',
        type=float,
        default=1e-3,
        help='Learning rate to use for training'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for the SGD optimizer'
    )
    parser.add_argument(
        '--num_mlp_layers', '-nmlp',
        type=int,
        default=10,
        help='Number of MLP layers to use if selected architecture contains MLP portion.'
    )
    parser.add_argument(
        '--action_type',
        type=str,
        default='absolute_action',
        choices=['delta_action', 'delta_action_norot', 'absolute_action', 'position', 'delta_position', 'position+gripper', 'delta_position+gripper', 'pose', 'delta_pose'],
        help='Type of action representation to use'
    )
    parser.add_argument(
        '--standardize_data',
        action='store_true'
    )
    parser.add_argument(
        '--coordinate_system',
        choices=['global', 'camera', 'disentangled'],
        default='disentangled'
    )
    parser.add_argument(
        '--loss',
        choices=['mse', 'cosine', 'cosine+mse'],
        default='cosine+mse'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='Path to a checkpoint file to resume training'
    )
    parser.add_argument(
        '--vMF_sample_method',
        type=str,
        default='rejection',
    )
    
    parser.add_argument(
        '--num_gpus', type=int, default=None,
        help='Number of GPUs to use (default: use all available)'
    )
    
    parser.add_argument(
        '--spatial_softmax', action='store_true',
        help='Use spatial softmax to extract features for ResNet architectures'
    )

    args = parser.parse_args()
    assert 128 % args.latent_dim == 0, "latent_dim must divide 128 evenly."

    if args.freeze_idm or args.freeze_fdm:
        assert 'aux' in args.architecture and 'latent_decoder' in args.architecture

    if 'latent_decoder' in args.architecture:
        assert args.idm_model_name != ''

        if 'aux' in args.architecture:
            assert args.fdm_model_name != ''
    
    if 'resnet' in args.architecture:
        assert args.resnet_layers_num != 0, "Choose either ResNet-18, 50, 101, 152, or 200"

    args.cameras = args.cameras.split(',')
    args.embodiments = args.embodiments.split(',')

    if args.valsets_path == '':
        args.valsets_path = args.datasets_path
        
    args.results_path = os.path.join(args.results_path, args.note)

    print('Arguments:', args) # Check argument correctness in jobs
    train(args)
