from pathlib import Path

MULTI_BACKBONE_PATH = (
    Path(__file__).parent / "models" / "TIMS_multi" / "Transfer_single_backbone.ckpt"
)

multi_config = {
    "model_name": "IM2DeepMulti",
    "batch_size": 16,
    "learning_rate": 0.0001,
    "AtomComp_kernel_size": 4,
    "DiatomComp_kernel_size": 2,
    "One_hot_kernel_size": 2,
    "AtomComp_out_channels_start": 256,
    "DiatomComp_out_channels_start": 128,
    "Global_units": 16,
    "OneHot_out_channels": 2,
    "Concat_units": 128,
    "AtomComp_MaxPool_kernel_size": 2,
    "DiatomComp_MaxPool_kernel_size": 2,
    "Mol_MaxPool_kernel_size": 2,
    "OneHot_MaxPool_kernel_size": 10,
    "LRelu_negative_slope": 0.1,
    "LRelu_saturation": 20,
    "L1_alpha": 0.00001,
    "delta": 0,
    "device": 0,
    "add_X_mol": False,
    "init": "normal",
    "backbone_SD_path": MULTI_BACKBONE_PATH,
}
