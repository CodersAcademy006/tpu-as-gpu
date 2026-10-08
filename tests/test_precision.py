from tpu_as_gpu.precision import autocast_dtype, compile_backend


def test_autocast_dtype_xla_is_bfloat16():
    assert autocast_dtype("xla") == "bfloat16"


def test_autocast_dtype_cuda_is_float16():
    assert autocast_dtype("cuda") == "float16"


def test_compile_backend_xla_is_openxla():
    assert compile_backend("xla") == "openxla"


def test_compile_backend_cuda_is_inductor():
    assert compile_backend("cuda") == "inductor"
