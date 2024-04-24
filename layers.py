import numpy as np


class Param:
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding, stride):
        

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )
        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.stride = stride

        self.X = None

    def pad(self, X_like):
        padding = self.padding
        batch_size, in_height, in_width, in_channels = X_like.shape

        in_height_pad = in_height + 2 * padding
        in_width_pad = in_width + 2 * padding

        X_pad = np.zeros((batch_size, in_height_pad, in_width_pad, in_channels))
        X_pad[:, padding:in_height + padding, padding:in_width + padding, :] = X_like

        return X_pad
    
    def unpad(self, X_pad_like):
        padding = self.padding
        _, in_height_pad, in_width_pad, _ = X_pad_like.shape

        in_height = in_height_pad - 2 * padding
        in_width = in_width_pad - 2 * padding

        return X_pad_like[:, padding:in_height + padding, padding:in_width + padding, :]

    def forward(self, X):
        self.X_pad = self.pad(X)
        batch_size, height, width, channels = self.X_pad.shape

        stride = self.stride

        filter_size = self.filter_size
        out_channels = self.out_channels

        out_height = height - filter_size + 1
        out_width = width - filter_size + 1
        
        out = np.zeros((batch_size, out_height, out_width, out_channels))

        for y in range(0, out_height, stride):
            for x in range(0, out_width, stride):
                X_window = self.X_pad[:, y:y+filter_size, x:x+filter_size, :]
                X_window_2d = X_window.reshape(batch_size, -1)
                W_2d = self.W.value.reshape(-1, out_channels)
                
                out[:, y, x, :] = np.dot(X_window_2d, W_2d) + self.B.value

        return out
