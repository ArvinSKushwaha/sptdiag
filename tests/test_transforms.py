from sptdiag import InertialFrame
import numpy as np


def test_inertial_frame():
    # test inertial frame
    inertial_frame = InertialFrame(0)
    inertial_frame.add(np.random.rand(1000, 2))
    assert np.allclose(inertial_frame.to_observer().data, inertial_frame.data)
