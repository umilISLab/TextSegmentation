import torch
import numpy as np
import pandas as pd
import regex as re
from collections import Counter
from itertools import permutations
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import KFold
from nltk import word_tokenize
from typing import Callable, Iterable, List, Union
import os


def import_data(path, **kwargs):
    data = pd.read_csv(path, sep = "\t", encoding = "utf-8", **kwargs)
    return data


def import_raw_corpus(folder, **kwargs):
    corpus = []
    names = []
    for file in os.listdir(folder):
        if not file.endswith(".txt"):
            continue
        names.append(file)
        with open(os.path.join(folder, file), "r", **kwargs) as f:
            corpus.append(f.read())
    return pd.Series(corpus, index = names)


def split_sentences(corpus, merge_short: int = 0):
    sentences = corpus.apply(lambda s: [(match.span()) for match in re.finditer(
        r"(?<!([Cc]o|[Nn]r|[Dd]ep|[Uu]d|[Aa]rtt?|ARTT?|[Ll]gs|[Rr]eg|[Dd]ott|[Dd]r|[Ss]igg?|[Pp]roc|[Aa]vv|[Uu]ff|[Ss]ez|[Ss]ent|[Cc]ass|[Cc]iv|[Ff]asc|[Dd]oc|[Vv]erb|[Pp]rot|[Pp]r|[Cc]od|[Cc]fr|[Pp]agg?|[Rr]el|\w\.\w|[A-Z]|www| [a-z]|[A-Z][a-z]))(\.|\n|;)(?!(\w\.|\d|it|eu|com|net|.{,5}\.))",
        s)])
    df = pd.concat([corpus, sentences], axis = 1)
    df.columns = ["text", "sep_boundaries"]

    df['text_length'] = df['text'].apply(len)
    df['sep_boundaries'] = df.apply(
        lambda row: [(0, 0)] + row['sep_boundaries'] + [(row['text_length'],)], axis=1)
    df['sentence_boundaries'] = df['sep_boundaries'].apply(
        lambda L: [(pre[-1], nxt[0]) for pre, nxt in zip(L, L[1:])])

    sent_df = df.explode("sentence_boundaries").reset_index()
    sent_df['start'] = sent_df['sentence_boundaries'].apply(lambda tup: tup[0])
    sent_df['end'] = sent_df['sentence_boundaries'].apply(lambda tup: tup[-1])
    sent_df['sent_text'] = sent_df.apply(lambda row: row['text'][row['start']:row['end']], axis=1)

    # Filter out NaN and empty rows
    sent_df = sent_df[sent_df['sent_text'].apply(lambda s: isinstance(s, str))]
    sent_df = sent_df[sent_df['sent_text'].apply(len) > 10]
    sent_df['sent_text'] = sent_df['sent_text'].apply(lambda s: s.strip())

    # Concatenate two sentences if too short
    if merge_short:
        prev = 0
        flag_prev = False
        for i,row in sent_df.iterrows():
            if row['end'] - row['start'] <= merge_short and flag_prev:
                sent_df.loc[prev, "end"] = sent_df.loc[i, "end"]
                sent_df.loc[prev, "sent_text"] = " ".join([sent_df.loc[prev, "sent_text"], sent_df.loc[i, "sent_text"]])
                sent_df = sent_df.drop(i)
                flag_prev = False
            elif row['end'] - row['start'] <= merge_short:
                flag_prev = True
            else:
                pass
            prev = i

    return sent_df


def exclude_partially_labeled(sent_df: pd.DataFrame, idfield: str, labelfield: str, min_labels: int, null_label: str, sep: str = ", "):

    sent_df[labelfield] = sent_df[labelfield].apply(lambda s: s.split(sep))
    grouped_df = sent_df.groupby(idfield).agg(list)
    labelsets = grouped_df[labelfield].apply(lambda L: set(sum(L, [])))
    to_drop = [idx for idx, labelset in labelsets.items() if len(labelset) <= min_labels and null_label in labelset]
    print("The following documents will be dropped:", to_drop)
    sent_df[labelfield] = sent_df[labelfield].apply(lambda L: sep.join(L))

    return sent_df[sent_df[idfield].apply(lambda idx: idx not in to_drop)]



def data_preprocessing(data, idfield, textfield, labelfield, sbert_vectors, synt_vectors, tfidf_vectors, null_label = ""):

    # Sentence position in its document
    data['sent_position'] = data.groupby(idfield).cumcount() + 1
    data['length'] = data[textfield].apply(lambda s: len(word_tokenize(s)))
    data['N_references'] = data[textfield].apply(lambda s: len(re.findall(r"(?<=([Cc]ass|[Aa]rtt?)\.?) *\d+", s)))
    data[textfield] = data[textfield].apply(lambda s: s.strip())

    # Binarizing labels
    mlb = MultiLabelBinarizer()
    data['Y'] = [v for v in mlb.fit_transform(data[labelfield].apply(lambda s: s.split(", ")))]
    data['Y'] /= data['Y'].apply(lambda v: sum(v))

    # Embeddings in the dataframe
    features = []
    if sbert_vectors:
        data['BERT'] = sbert_vectors
        features.append("BERT")
    if synt_vectors:
        data['Synt'] = synt_vectors
        data['Synt'] = data.apply(lambda row: dict(sent_position = row['sent_position'], length = row['length'], refs = row['N_references'], **row['Synt']), axis = 1)
        features.append("Synt")
    if tfidf_vectors:
        data['TFIDF'] = tfidf_vectors
        features.append("TFIDF")

    data['feature_dicts'] = data.apply(lambda row: {k: v for feat in features for k, v in row[feat].items()},
                                       axis=1)

    grouped_data = data.groupby(idfield).agg(list)

    # Remove documents with a majority of unlabeled sentences
    if null_label:
        doc_label_counts = grouped_data[labelfield].apply(Counter)
        to_remove = [id_ for id_, count in doc_label_counts.items() if count[null_label] == max(count.values())]
        grouped_data.drop(to_remove, inplace = True)

    return grouped_data, list(mlb.classes_)


def batch_generator(data: list, batch_size: int, transform: Callable = None):
    """
    Utility to generate batches.

    Args:
        sentences (List[graph.Sentence]): List of LAZY sentences.
        batch_size (int): Number of sentences per batch.
        transform (Callable): Function to transform each sentence.

    Returns:
        Generator.
    """
    N = len(data)

    for i in range(0, N, batch_size):
        batch = data[i:i + batch_size]
        if transform is None:
            yield batch
        else:
            yield batch, list(map(transform, batch))

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


# Getting Sentence Embeddings
def sentence_embeddings(sentences, tokenizer, model, max_length=512, pooling: bool = True):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input.to("cuda"))

    # Perform pooling. In this case, mean pooling
    if pooling:
        final = mean_pooling(model_output, encoded_input['attention_mask']).detach().cpu().numpy()
    else:
        final = model_output.last_hidden_state[:, 0, :].detach().cpu().numpy()

    return final


# Defining additional features
def additional_features(sentences, tfidf):
    tfidf.fit(sentences)
    return tfidf.transform(sentences), tfidf.get_feature_names_out()


def doc_fold_mapping(data, idfield, labelfield, n_folds):
    kfold_splitter = KFold(n_splits=n_folds, shuffle=True)
    doc_ids = data[idfield].unique()
    most_frequent_label = data[labelfield].value_counts().idxmax()
    strat_variable = [data.loc[data[idfield] == idx, labelfield].value_counts()[most_frequent_label] for idx in doc_ids]
    bins = np.linspace(min(strat_variable), max(strat_variable), n_folds)
    stratification = np.digitize(strat_variable, bins = bins)
    folds = {doc_ids[idx]: i for i, (train, test) in enumerate(kfold_splitter.split(doc_ids, stratification)) for idx in test}
    return folds


def pad_sequence_2d(tensors):
    shapes = list(map(lambda t: t.shape, tensors))
    dim1, dim2 = list(zip(*shapes))
    dim1max = max(dim1)
    dim2max = max(dim2)

    masks = list(map(lambda t: (dim1max - t.shape[0], dim2max - t.shape[1]), tensors))
    padded_tensors = [torch.nn.functional.pad(t, (0, m[1], 0, m[0]), "constant", 0) for t, m in zip(tensors, masks)]
    return padded_tensors


def map_prob_labels(Y: Union[np.ndarray, torch.Tensor], labels: Iterable[Union[str, int]], dim: int = -1):
    """
    Mapping probability (or score) arrays into actual labels by taking the argmax.

    :param Y: Probability array.
    :param labels: List of labels.
    :param dim: Dimension along which to compute the argmax.
    :return: List of labels.
    """
    idx = Y.argmax(axis = dim).flatten()

    return [labels[k] for k in idx]


def map_multilabel(y_pred, y_true, sep = ", "):
    splitted = y_true.split(sep)
    if len(splitted) <= 1:
        return y_true
    elif y_pred in splitted:
        return y_pred
    else:
        return splitted[0]


def tensor_check(y):
    if not isinstance(y, torch.Tensor):
        T = torch.from_numpy(np.array(y))

    return T


def train_break_rules(losses: List[float], tol: float, break_seq: int):

    latest_deltas = [(l1 - l2) / l1 for l1,l2 in zip(losses[-break_seq-1:], losses[-break_seq:])]

    break_rules = [
        all([d <= tol for d in latest_deltas]),
        all([l <= tol for l in losses[-break_seq:]])
    ]

    return any(break_rules)


def extract_features_from_nodes(Gs):
    X = list(map(
        lambda G:
        torch.from_numpy(np.array([
            list(G.nodes[n].values())
            for n in G.nodes])),
        Gs
    ))
    return X


def maximum_index_alignment(M):

    best_alignment = None
    best_score = 0
    # Greedy way
    indices = list(range(M.shape[0]))
    for tup in permutations(indices):
        alignment = list(zip(tup, indices))
        new_score = sum([M[i,j] for i,j in alignment])
        if new_score >= best_score:
            best_score = new_score
            best_alignment = alignment

    alignment_matrix = torch.zeros(M.shape)
    for i,j in best_alignment:
        alignment_matrix[i,j] = 1

    return alignment_matrix


def random_init_tensor_list(shapes):

    output = []
    for s in shapes:
        T = torch.rand(s)
        T = torch.nn.functional.normalize(T, p=1, dim=-1)
        output.append(T)

    return output


def onehot_tensor(shape, position):

    T = torch.zeros(shape)
    T[position] = 1

    return T