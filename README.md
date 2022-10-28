# active_learning
Active Learning -- probbaly mostly - LIGHTLY 

```python

  (projection_head): SimSiamProjectionHead(
    (layers): Sequential(
      (0): Linear(in_features=512, out_features=512, bias=False)
      (1): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Linear(in_features=512, out_features=512, bias=False)
      (4): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU()
      (6): Linear(in_features=512, out_features=512, bias=False)
      (7): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    )
  )
  (prediction_head): SimSiamPredictionHead(
    (layers): Sequential(
      (0): Linear(in_features=512, out_features=128, bias=False)
      (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Linear(in_features=128, out_features=512, bias=True)

```
