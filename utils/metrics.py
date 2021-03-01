from abc import abstractmethod
from collections import defaultdict


class MetricBase(object):

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_metric(self, reset=True):
        raise NotImplemented

    def __call__(self, p_ids, pred, eval_file):
        return self.evaluate(p_ids, pred, eval_file)


# def _bmeso_tag_to_spans(tags, ignore_labels=None):
#     """
#     给定一个tags的lis，比如['O', 'B-singer', 'M-singer', 'E-singer', 'O', 'O']。
#     返回[('singer', (1, 4))] (左闭右开区间)
#
#     :param tags: List[str],
#     :param ignore_labels: List[str], 在该list中的label将被忽略
#     :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
#     """
#     ignore_labels = set(ignore_labels) if ignore_labels else set()
#
#     spans = []
#     prev_bmes_tag = None
#     for idx, tag in enumerate(tags):
#         tag = tag.lower()
#         bmes_tag, label = tag[:1], tag[2:]
#         if bmes_tag in ('b', 's'):
#             spans.append((label, [idx, idx]))
#         elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
#             spans[-1][1][1] = idx
#         elif bmes_tag == 'o':
#             pass
#         else:
#             spans.append((label, [idx, idx]))
#         prev_bmes_tag = bmes_tag
#     return [(span[0], (span[1][0], span[1][1] + 1))
#             for span in spans
#             if span[0] not in ignore_labels
#             ]

def _bmeso_tag_to_spans(tags,text, ignore_labels=None):
    r"""
    给定一个tags的lis，比如['O', 'B-singer', 'M-singer', 'E-singer', 'O', 'O']。
    返回[('singer', (1, 4))] (左闭右开区间)

    :param tags: List[str],
    :param ignore_labels: List[str], 在该list中的label将被忽略
    :return: List[Tuple[str, List[int, int]]]. [(label，[start, end])]
    """
    ignore_labels = set(ignore_labels) if ignore_labels else set()

    spans = []
    prev_bmes_tag = None
    for idx, tag in enumerate(tags):
        tag = tag.lower()
        bmes_tag, label = tag[:1], tag[2:]
        if bmes_tag in ('b', 's'):
            spans.append((label, [idx, idx]))
        elif bmes_tag in ('m', 'e') and prev_bmes_tag in ('b', 'm') and label == spans[-1][0]:
            spans[-1][1][1] = idx
        elif bmes_tag == 'o':
            pass
        else:
            spans.append((label, [idx, idx]))
        prev_bmes_tag = bmes_tag

    ent_lst = []
    for span in spans:
        ent_type=span[0]
        start=span[1][0]
        end=span[1][1]
        if start == 1 and end > len(text):
            continue
        ent_name = text[start - 1:end]
        ent_lst.append((start-1, end-1, ent_name, ent_type.upper()))


    return ent_lst

class SpanFPreRecMetric(MetricBase):
    def __init__(self, tag_type, pred=None, target=None, encoding_type='bmeso',
                 only_gross=True, f_type='micro', beta=1):
        self.tag_type = tag_type
        self.only_gross = only_gross
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta ** 2
        self.encoding_type = encoding_type
        if self.encoding_type == 'bmeso':
            self.tag_to_span_func = _bmeso_tag_to_spans

        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)

    def evaluate(self, p_ids, preds, eval_file):
        answer_dict = {}
        for p_id, pred in zip(p_ids.tolist(), preds.tolist()):
            gold_ = eval_file[p_id].gold_answer
            pred_ = [self.tag_type[tag] for tag in pred]
            # gold_str_tags = [self.tag_type[tag] for tag in gold]
            pred_spans = self.tag_to_span_func(pred_)
            gold_spans = self.tag_to_span_func(gold_)
            answer_dict[str(p_id)] = [pred_spans, gold_spans]
            for span in pred_spans:
                if span in gold_spans:
                    self._true_positives[span[0]] += 1
                    gold_spans.remove(span)
                else:
                    self._false_positives[span[0]] += 1
            for span in gold_spans:
                self._false_negatives[span[0]] += 1
        return answer_dict

    def get_metric(self, reset=True):
        """get_metric函数将根据evaluate函数累计的评价指标统计量来计算最终的评价结果."""
        evaluate_result = {}
        if not self.only_gross or self.f_type == 'macro':
            tags = set(self._false_negatives.keys())
            tags.update(set(self._false_positives.keys()))
            tags.update(set(self._true_positives.keys()))
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                tp = self._true_positives[tag]
                fn = self._false_negatives[tag]
                fp = self._false_positives[tag]
                f, pre, rec = self._compute_f_pre_rec(tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':  # tag!=''防止无tag的情况
                    f_key = 'f-{}'.format(tag)
                    pre_key = 'pre-{}'.format(tag)
                    rec_key = 'rec-{}'.format(tag)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum / len(tags)
                evaluate_result['pre'] = pre_sum / len(tags)
                evaluate_result['rec'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            f, pre, rec,em,pre_num,gold_num = self._compute_f_pre_rec(sum(self._true_positives.values()),
                                                  sum(self._false_negatives.values()),
                                                  sum(self._false_positives.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec
            evaluate_result['em'] = em
            evaluate_result['pre_num'] = pre_num
            evaluate_result['gold'] = gold_num


        if reset:
            self._true_positives = defaultdict(int)
            self._false_positives = defaultdict(int)
            self._false_negatives = defaultdict(int)

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)
        print(evaluate_result)
        return evaluate_result

    def _compute_f_pre_rec(self, tp, fn, fp):
        """

        :param tp: int, true positive
        :param fn: int, false negative
        :param fp: int, false positive
        :return: (f, pre, rec)
        """
        pre = tp / (fp + tp + 1e-13)
        rec = tp / (fn + tp + 1e-13)
        f = (1 + self.beta_square) * pre * rec / (self.beta_square * pre + rec + 1e-13)


        return f, pre, rec,tp,fp + tp,fn + tp
