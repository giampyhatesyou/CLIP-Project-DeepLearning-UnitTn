# ProtoCoCoOp: base-to-novel generalisation with CLIP

Deep Learning course’s Project at the University of Trento. We worked on the base-to-novel problem with CLIP. When you adapt CLIP to a set of classes it usually gets better on those (the base classes) but loses its zero-shot strength on the new ones (the novel classes). The goal is to keep both.

Starting from CoOp and CoCoOp, two prompt-learning methods, and added two features. The first is knowledge distillation: we keep the frozen CLIP as a teacher so the adapted model does not drift too far from its original zero-shot behaviour. The second is class prototypes, which consists in building from CLIP image features and fused at inference time to bring back accuracy on the base classes. We call the combination ProtoCoCoOp.

We evaluated every method by the harmonic mean between base and novel accuracy, because it let us consider both for a fair tradeoff.

## Results on Flowers102

| Method | Base | Novel | Harmonic mean |
| - | - | - | - |
| CLIP zero-shot | 71.33 | 78.24 | 74.62 |
| CoCoOp | 93.57 | 75.19 | 83.38 |
| CoCoOp + KD | 87.18 | 76.47 | 81.47 |
| ProtoCoCoOp (ours) | 93.85 | 76.74 | 84.44 |


ProtoCoCoOp reached the best harmonic mean, 84.44, and the best base accuracy, 93.85, among the methods we tried.

## Stack

Python, PyTorch, CLIP, prompt learning (CoOp and CoCoOp), knowledge distillation. Everything runs from a single notebook, `dl\_project.ipynb`, which holds the Flowers102 data, the CLIP baseline, our method and the plots.

## Team

Group project for the Deep Learning course. We split the work fairly evenly across the team.

## References

- Radford et al., Learning Transferable Visual Models From Natural Language Supervision (CLIP), 2021.

- Zhou et al., Conditional Prompt Learning for Vision-Language Models (CoCoOp), CVPR 2022.

