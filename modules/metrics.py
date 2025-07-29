from pycocoevalcap.bleu.bleu import Bleu
# from pycocoevalcap.meteor import Meteor  # METEOR disabled due to Java issues
from pycocoevalcap.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def compute_scores(gts, res):
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        # (Meteor(), "METEOR"),  # Disabled to avoid BrokenPipeError
        (Rouge(), "ROUGE_L"),
        (Cider(), "Cider")
    ]
    eval_res = {}
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res)
        except Exception as e:
            print(f"Error in {method}: {e}")
            continue
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res
