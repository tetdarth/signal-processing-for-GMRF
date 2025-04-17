import torch
import torch.nn.functional as F

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.feature = None
        self.feature_grad = None

        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.feature_grad = grad_output[0].detach()
        
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class=None, size=50):
        """ 
        Grad-CAMのヒートマップを生成

        Args:
            input_tensor(pytorch.tensor) : 入力データ
            target_class(int) : 入力データの分類クラス
            size(int) : sizeに指定した数字に引き延ばす
        Returs:
            cam_interp : Grad-CAMの結果
        """
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        loss = output[0, target_class]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        weights = self.feature_grad.mean(dim=2, keepdim=True)  # 平均: (N, C, 1)
        cam = (weights * self.feature).sum(dim=1)  # 重み付き和: (N, L)
        cam = F.relu(cam)

        cam = cam - cam.min(dim=1, keepdim=True)[0]
        cam = cam / (cam.max(dim=1, keepdim=True)[0] + 1e-8)

        cam_interp = F.interpolate(cam.unsqueeze(1), size=size, mode='linear', align_corners=False)
        cam_interp = cam_interp.squeeze(1)

        return cam_interp  # shape: (size, L)

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
