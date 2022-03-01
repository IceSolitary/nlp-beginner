import collections
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
import torch


class Poem_Dataset(Dataset):
    def __init__(self, file_name):
        self.start_token = "B"
        self.end_token = "E"
        self.pad_token = "P"

        self.poem_vector, self.word_int_map, self.words = self.process_poems2(file_name, 24)

    def __len__(self):
        return len(self.poem_vector)

    def __getitem__(self, idx):
        return self.poem_vector[idx]

    def collate_fn(self, batch_data):

        return batch_data

    def process_poems2(self, file_name, max_len):
        """
        :param file_name:
        :return: poems_vector  have tow dimmention ,first is the poem, the second is the word_index
        e.g. [[1,2,3,4,5,6,7,8,9,10],[9,6,3,8,5,2,7,4,1]]

        """
        poems = []
        with open(file_name, "r", encoding='utf-8', ) as f:
            # content = ''
            for line in f.readlines():
                try:
                    line = line.strip()
                    if line:
                        content = line.replace(' '' ', '').replace('，', '').replace('。', '')
                        if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                self.start_token in content or self.end_token in content:
                            continue
                        if len(content) < 5 or len(content) > max_len-2:
                            continue
                        content = self.start_token + content + self.end_token
                        # if len(content) < max_len:
                        #     for index in range(max_len - len(content)):
                        #         content = content + self.pad_token
                        poems.append(content)
                        # content = ''
                except ValueError as e:
                    # print("error")
                    pass
        # 按诗的字数排序
        poems = sorted(poems, key=lambda line: len(line))
        # print(poems)
        # 统计每个字出现次数
        all_words = []
        for poem in poems:
            all_words += [word for word in poem]
        counter = collections.Counter(all_words)  # 统计词和词频。
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # 排序
        words, _ = zip(*count_pairs)
        words = words[:len(words)] + (' ',)
        word_int_map = dict(zip(words, range(len(words))))
        poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
        return poems_vector, word_int_map, words
