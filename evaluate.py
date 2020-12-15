import torch
import random
from train import indexesFromSentence
from load import SOS_token, EOS_token
from load import MAX_LENGTH, loadPrepareData, Voc
from model import *
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        # return mean of sentence_score
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                terminates.append(([voc.index2word[idx.item()] for idx in self.sentence_idxes] + ['<EOS>'],
                                   self.avgScore())) # tuple(word_list, score_float
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<EOS>')
            else:
                words.append(voc.index2word[self.sentence_idxes[i].item()])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<EOS>')
        return (words, self.avgScore())


def beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size, max_length=MAX_LENGTH):
    terminal_sentences, prev_top_sentences, next_top_sentences = [], [], []
    prev_top_sentences.append(Sentence(decoder_hidden))
    for i in range(max_length):
        for sentence in prev_top_sentences:
            decoder_input = torch.LongTensor([[sentence.last_idx]])
            decoder_input = decoder_input.to(device)

            decoder_hidden = sentence.decoder_hidden
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            topv, topi = decoder_output.topk(beam_size)
            term, top = sentence.addTopk(topi, topv, decoder_hidden, beam_size, voc)
            terminal_sentences.extend(term)
            next_top_sentences.extend(top)

        next_top_sentences.sort(key=lambda s: s.avgScore(), reverse=True)
        prev_top_sentences = next_top_sentences[:beam_size]
        next_top_sentences = []

    terminal_sentences += [sentence.toWordScore(voc) for sentence in prev_top_sentences]
    terminal_sentences.sort(key=lambda x: x[1], reverse=True)

    n = min(len(terminal_sentences), 15)
    return terminal_sentences[:n]


def decode(decoder, decoder_hidden, encoder_outputs, voc, k, p, max_length=MAX_LENGTH):
    # Greedy search: k = 1, p = 0. Top-K sampling: k > 1, p = 0. Nucleus sampling: k = inf, 0 < p < 1

    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_input = decoder_input.to(device)

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        topv, topi = decoder_output.topk(k)

        if p > 0:
            cumulative_prob = topv.cumsum(dim=1)
            idxes = cumulative_prob < p
            topv, topi = topv[idxes], topi[idxes]
            ni = topi.multinomial(topi)
        else:
            ni = topi[0][0]

        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(voc.index2word[ni.item()])

        decoder_input = torch.LongTensor([[ni]])
        decoder_input = decoder_input.to(device)

    return decoded_words, decoder_attentions[:di + 1]


def evaluate(encoder, decoder, voc, sentence, beam_size, k, p, max_length=MAX_LENGTH, hidvar=0):
    indexes_batch = [indexesFromSentence(voc, sentence)]  # [1, seq_len]
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)

    encoder_outputs, encoder_hidden = encoder(input_batch, lengths, None)  # tutorial: instead in class of decoder

    if hidvar:
        z = Variable(torch.randn([4, 1, hidvar.z_hidden_size]))
        z = z.to(device)
        decoder_hidden = torch.cat([encoder_hidden[:decoder.n_layers], z[:decoder.n_layers]], 2)
    else:
        decoder_hidden = encoder_hidden[:decoder.n_layers]

    if beam_size == 1:
        return decode(decoder, decoder_hidden, encoder_outputs, voc, k, p)
    else:
        return beam_decode(decoder, decoder_hidden, encoder_outputs, voc, beam_size)


def evaluateInput(encoder, decoder, voc, beam_size, k, p):
    pair = ''
    while(1):
        try:
            pair = input('> ')
            if pair == 'q': break
            if beam_size == 1:
                output_words, _ = evaluate(encoder, decoder, voc, pair, beam_size, k, p)
                output_sentence = ' '.join(output_words)
                print('<', output_sentence)
            else:
                output_words_list = evaluate(encoder, decoder, voc, pair, beam_size, k, p)
                for output_words, score in output_words_list:
                    output_sentence = ' '.join(output_words)
                    print("{:.3f} < {}".format(score, output_sentence))

        except KeyError:
            print("Error: Encountered unknown word.")

def evaluateScore(encoder, decoder, voc, beam_size, k, p, hidvar=0):
    gram1_bleu_score = []
    gram2_bleu_score = []

    for i in range(0, len(testpairs), 1):
        sentence = testpairs[i][0]
        reference = testpairs[i][1:]
        templist = []

        for k in range(len(reference)):
            if (reference[k] != ''):
                temp = reference[k].split(' ')
                templist.append(temp)

        sentence = normalizeString(sentence)
        output_words = evaluate(encoder, decoder, voc, sentence, beam_size, k, p, hidvar)
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        chencherry = SmoothingFunction()

        score1 = sentence_bleu(templist, output_words, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
        score2 = sentence_bleu(templist, output_words, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)

        gram1_bleu_score.append(score1)
        gram2_bleu_score.append(score2)
        if i % 1000 == 0:
            print(i, sum(gram1_bleu_score) / len(gram1_bleu_score), sum(gram2_bleu_score) / len(gram2_bleu_score))
    print("Total Bleu Score for 1 grams on testing pairs: ", sum(gram1_bleu_score) / len(gram1_bleu_score))
    print("Total Bleu Score for 2 grams on testing pairs: ", sum(gram2_bleu_score) / len(gram2_bleu_score))



def runTest(n_layers, hidden_size, reverse, modelFile, attn_model, beam_size, k, p, v, inp, corpus):
    torch.set_grad_enabled(False)

    voc, pairs = loadPrepareData(corpus)
    embedding = nn.Embedding(voc.num_words, hidden_size)

    if v:
        embedding_decoder = nn.Embedding(voc.num_words, hidden_size * 2)
        encoder = EncoderRNN(hidden_size, embedding, n_layers)
        decoder = DecoderRNN(attn_model, embedding_decoder, hidden_size * 2, voc.num_words, n_layers)
        hidvar = LatentVariation(hidden_size * 2, hidden_size)

        hidvar.load_state_dict(checkpoint['hv'])
        hidvar = hidvar.to(device)
    else:
        encoder = EncoderRNN(voc.num_words, hidden_size, embedding, n_layers)
        attn_model = attn_model
        decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.n_words, n_layers)

    checkpoint = torch.load(modelFile)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    # train mode set to false, effect only on dropout, batchNorm
    encoder.train(False);
    decoder.train(False);

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    if inp:
        evaluateInput(encoder, decoder, hidvar, voc, beam_size, k, p, hidvar)
    else:
        evaluateScore(encoder, decoder, hidvar, voc, pairs, reverse, beam_size, hidvar)