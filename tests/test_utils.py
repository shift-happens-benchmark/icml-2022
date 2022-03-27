from shifthappens import utils as sh_utils


def test_dict_product():
    dict_values = dict(a=(0, 1), b=(2, 3))
    prod_values = list(sh_utils.dict_product(dict_values))
    assert len(prod_values) == 2*2
    for it in prod_values:
        assert isinstance(it["a"], int)
        assert isinstance(it["b"], int)
