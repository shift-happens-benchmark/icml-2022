from pkg_resources import resource_filename
from shifthappens import benchmark as sh_benchmark

osr_split_path = resource_filename(__name__, 'imagenet_osr_splits_winter21.pkl')
@sh_benchmark.register_task(
    name="SSB", relative_data_folder="ssb", standalone=True
)