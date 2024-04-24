import numpy as np

from layers import ConvolutionalLayer


def main():
    X = np.array([
              [
               [[1.0], [2.0]],
               [[0.0], [-1.0]]
              ]
              ,
              [
               [[0.0], [1.0]],
               [[-2.0], [-1.0]]
              ]
             ])

    layer = ConvolutionalLayer(in_channels=1, out_channels=1, filter_size=2, padding=0, stride=1)

    layer.W.value = np.zeros_like(layer.W.value)
    layer.W.value[0, 0, 0, 0] = 1.0
    layer.B.value = np.ones_like(layer.B.value)

    result = layer.forward(X)

    assert result.shape == (2, 1, 1, 1)
    assert np.all(result == X[:, :1, :1, :1] +1), "result: %s, X: %s" % (result, X[:, :1, :1, :1])

    layer = ConvolutionalLayer(in_channels=1, out_channels=2, filter_size=2, padding=0, stride=1)
    result = layer.forward(X)
    assert result.shape == (2, 1, 1, 2)

    X = np.array([
              [
               [[1.0, 0.0], [2.0, 1.0]],
               [[0.0, -1.0], [-1.0, -2.0]]
              ]
              ,
              [
               [[0.0, 1.0], [1.0, -1.0]],
               [[-2.0, 2.0], [-1.0, 0.0]]
              ]
             ])
    
    layer = ConvolutionalLayer(in_channels=2, out_channels=2, filter_size=2, padding=0, stride=1)
    result = layer.forward(X)
    assert result.shape == (2, 1, 1, 2)


if __name__ == "__main__":
    main()
