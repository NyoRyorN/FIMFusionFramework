# beforefusionGrad-CAM
# multipleaction task fusion Class Activation Mapping
# spatial(T, V) visualize and integrateprocessing

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAM_SkeletonPreFusion(GradCAM):
    """beforefusionGrad-CAMclass

    multipleaction task beforefusion(Pre-Fusion) Grad-CAM.
    Grad-CAM, spatial(T×V)
    visualize and multipleaction task(A) integrateprocessing.

    Key characteristics:
    - input shape (N,A,C,T,V)
    - output shape (N,A,T,V) visualize
    - multipleaction task processing
    - temporal×joints 2generate

    Args:
        model (torch.nn.Module): target and neural network
        target_layers (list): Grad-CAM applytarget list
        reshape_transform (callable, optional): function
    """

    def __init__(self, model: torch.nn.Module, target_layers: list, reshape_transform: callable = None):
        # parentclass(GradCAM) initialize
        super(GradCAM_SkeletonPreFusion, self).__init__(
            model, target_layers, reshape_transform)

    def get_target_width_height(self, input_tensor: torch.Tensor):
        """target / getfunction

        spatial CAM resolve.
        temporalframe(T), joints(V) as set,
        (T,V) 2 as visualize row.

        Args:
            input_tensor (torch.Tensor): input tensor (N,A,C,T,V)

        Returns:
            tuple: (width, height) = (V, T) combination
        """
        # input: (N,A,C,T,V).output (N,A,T,V) return.
        # CAM (T,V) (W=V, H=T) return
        T = input_tensor.size(-2)  # temporalframe
        V = input_tensor.size(-1)  # joints
        return V, T  # (width, height)

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        """multiple CAM integratefunction

        multipleaction task(A) integrateprocessing.
        in, eachaction task CAM return and,
        per action visualize.

        Args:
            cam_per_target_layer (list): eachtarget CAMlist, eachelement (N,1,T,V)

        Returns:
            np.ndarray: integrateCAMcolumn (N,A,T,V)

        Note:
            - scale_cam_image＆ReLU
            - A(action task) CAM
        """
        # , Here A (=) return
        # cam_per_target_layer List[np.ndarray], eachelement (N,1,T,V)
        cams = np.concatenate(cam_per_target_layer, axis=1)  # (N,A,T,V)
        return cams  # scale_cam_image ＆ReLU
