# BlackSwan
Being Right when it Really Matters. #NLP #transferLearning


This is sanitized code (due to NDA) from a project for a startup. 

## What the code does:
Allows you to customize "off the shelf embeddings" via transfer learning training tasks.

The custom_loss function allows you to pass in an array of any size and apply asymetric weights for misclassification. For example: given the weight matrix 
```
INIT_COST_WEIGHTS = np.ones((3,3))
INIT_COST_WEIGHTS[1,0]=5
INIT_COST_WEIGHTS[2,0]=15
INIT_COST_WEIGHTS[2,1]=1
```
We are applying a 15x penalty to anything that classified to class 0 from true class 2. 

## In my case: 
Word2Vec was customized via a multilabel/multficlass classification problem to predict "tags/topics" in an email corpus.

The customized embeddings was then used to improve perfromance on a different classification task using the same corpus.
