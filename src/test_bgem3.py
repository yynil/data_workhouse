if __name__ == '__main__':
    bge_path = '/media/yueyulin/KINGSTON/models/bge-m3'
    from transformers import AutoTokenizer, AutoModel
    import torch
    # Sentences we want sentence embeddings for
    sentences = ['我打算取消订单','我要取消订单','我要退货','我要退款']

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained(bge_path)
    model = AutoModel.from_pretrained(bge_path)
    model.eval()

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # for s2p(short query to long passage) retrieval task, add an instruction to query (not add instruction for passages)
    # encoded_input = tokenizer([instruction + q for q in queries], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    print("Sentence embeddings:", sentence_embeddings)

    from sentence_transformers.util import pairwise_cos_sim
    for qid in range(len(sentences)):
        query = sentence_embeddings[qid]
        for i in range(len(sentences)):
            if i != qid:
                print(f'{sentences[qid]} vs {sentences[i]} is {pairwise_cos_sim(query.unsqueeze(0),sentence_embeddings[i].unsqueeze(0))}')

        print('-----------------------')


    from FlagEmbedding import BGEM3FlagModel
    model = BGEM3FlagModel(bge_path,device='cuda')
    sentence_embeddings = model.encode(sentences,batch_size=2,max_length=2048)['dense_vecs']
    print("Sentence embeddings:", sentence_embeddings)
    sentence_embeddings = torch.tensor(sentence_embeddings)
    for qid in range(len(sentences)):
        query = sentence_embeddings[qid]
        for i in range(len(sentences)):
            if i != qid:
                print(f'{sentences[qid]} vs {sentences[i]} is {pairwise_cos_sim(query.unsqueeze(0),sentence_embeddings[i].unsqueeze(0))}')

        print('-----------------------')