import logging

from transformers import BertTokenizer, AlbertModel, AlbertPreTrainedModel, AlbertForSequenceClassification

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

tokenizer = BertTokenizer.from_pretrained("transformer_cpt/albert_chinese_large/")
a = tokenizer.tokenize('dsfwe我爱北京天安门')
print(a)


class Test(AlbertPreTrainedModel):
    """
    ERENet : entity relation jointed extraction
    """

    def __init__(self, config):
        super(Test, self).__init__(config)
        # BERT model
        self.albert = AlbertModel(config)
        pass


# AlbertForSequenceClassification.from_pretrained('albert-base-v2')
model = Test.from_pretrained("transformer_cpt/albert_chinese_large/")
