In this directory, we provide the prediction results of our model on the test sets.

It includes the results for both tail entity prediction (forward direction) and head entity prediction (backward direction),
and the evaluation metrics values (MRR, Hits@{1,3,10}).

To illustrate the meaning of each field, take the following as an example:

```json
{
  "head": "Blank, Blank, Blank",
  "relation": "performer",
  "tail": "Contrived",
  "pred_tail": "cauda pavonis",
  "pred_score": 0.5772,
  "topk_score_info": "{\"cauda pavonis\": 0.577, \"contrive\": 0.568, \"Magrudergrind\": 0.557}",
  "rank": 63,
  "correct": false
}
```

`head`: head entity

`relation`: relation text (for head entity prediction, it starts with `inverse`)

`tail`: the groundtruth tail entity

`pred_tail`: predicted tail entity by our model

`pred_score`: the cosine similarity score for the predicted entity

`topk_score_info`: top-3 predicted entities and cosine similarity scores

`rank`: the rank for the groundtruth tail entity (start from 1)

`correct`: whether the prediction is correct or not, this is set to true only if the `pred_tail` and `tail` are the same.
