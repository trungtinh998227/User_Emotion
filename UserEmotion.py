# -*- coding: utf-8 -*-
import csv
import pandas as pd
import requests
import os.path
from bs4 import BeautifulSoup
from pyvi import ViTokenizer
import glob
from string import punctuation
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

"""
        Lấy dữ liệu từ mục "giao thông" của trang web
        1. Lấy link mục giao thông #Done# 
        2. Truy cập vào link ---> Lấy link mục 'Quản lí'-'Hạ tầng'-'Vận tải' #Done#
        3. Từng link, lấy nội dung theo từng bài lưu về các thư mục 'Quản lí'-'Hạ tầng'-'Vận tải' tương ứng #Done#
        4. clean data --> tách từ, loại bỏ stop word -> lưu về thư mục 'Quản lí_clean'-'Hạ tầng_clean'-'Vận tải_clean'
        5. Huấn luyện dữ liệu --> xác định nội dung chính của bài báo thuộc về loại tích cực hay tiêu cực
        6. Độ chính xác --> tính độ chính xác của dữ liệu training
        7. 
"""


# Hàm dùng để raw dữ liệu từ trang web
def get_soup(link):
    return BeautifulSoup(link.content, 'lxml')
    pass


def get_articles(soup_input):
    art = []
    for h in soup_input.find_all(['h2', 'h3']):
        if h.find('a', attrs={"href": "javascript:void(0)"}) or h.find('a', attrs={"href": "javascript:void(0);"}):
            continue
        art.append(h.find('a', href=True)['href'])
    return art
    pass


def get_data(art_link):
    request = requests.get(art_link)
    soup_data = get_soup(request)
    content = ""
    title = soup_data.find('h1').find('div').text
    subtitle = soup_data.find('h2').find('div').text
    dateArt = soup_data.find(class_="dateArt").text

    for div in soup_data.find_all('div', class_="bodyArt"):
        for p in div.find_all('p'):
            if p.find_all('span'):
                for span in p.find_all('span'):
                    if span.find_all('span'):
                        for span_child in span.find_all('span'):
                            content += span_child.text
                    else:
                        content += span.text
                pass
            else:
                content += p.text
                content += "\n"
    return title, subtitle, dateArt, content
    pass


def write_data_raw(title, subtitle, date, content, document):
    path = './DataRaw/' + document + '/'
    try:
        os.makedirs('DataRaw/' + document)
    except FileExistsError:
        pass
    num_files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
    f = open(path + str(num_files) + "_raw.txt", "w+", encoding='utf-8')
    f.write(title + '\n')
    f.write(subtitle + '\n')
    f.write(date + '\n')
    f.write(content)
    f.close()
    pass


def assign_label_data(folder):
    positive_word = []
    negative_word = []
    with open("./Emotionword/vietnamese_negative_words.txt", encoding="utf-8") as f:
        text = f.read()
        for word in text.split():
            negative_word.append(word)
        f.close()
    with open("./Emotionword/vietnamese_positive_words.txt", encoding="utf-8") as f2:
        text = f2.read()
        for word in text.split():
            word.replace(" ", "_")
            positive_word.append(word)
        f2.close()
    paths = glob.glob("./DataClean/" + folder + "_clean/*.txt")
    for path in paths:
        neg = 0
        pos = 0
        with open(path, encoding="utf-8") as file:
            text = file.read()
            for word in text.split():
                if word in negative_word:
                    neg += 1
                if word in positive_word:
                    pos += 1
        try:
            os.makedirs('DataLabel/' + folder)
        except FileExistsError:
            pass
        num_files = len([f for f in os.listdir('./DataLabel/' + folder + '/') if
                         os.path.isfile(os.path.join('./DataLabel/' + folder + '/', f))])
        f = open('./DataLabel/' + folder + '/' + str(num_files) + "_label.txt", "w+")
        if neg > pos:
            f.write('Negative')
        else:
            f.write('Positive')
        f.close()


def clean_dat_text(text_input):
    text_lower = text_input.lower()
    text_token = ViTokenizer.tokenize(text_lower)
    stop_word_ = []
    with open("./Stopword/vietnamese-stopwords.txt", encoding="utf-8") as f:
        t = f.read()
        for word in t.split():
            stop_word_.append(word)
        f.close()
    punc_ = list(punctuation)
    stop_word = stop_word_ + punc_
    sentences = []
    sent = []
    for word in text_token.split(" "):
        if word not in stop_word:
            if "_" in word or word.isalpha() is True:
                sent.append(word)
    sentences.append(" ".join(sent))
    return sentences


def load_and_clean_data(doc):
    paths = glob.glob("./DataRaw/" + doc + "/*.txt")
    data = []
    for path in paths:
        with open(path, encoding="utf-8") as file:
            text = file.read()
            text_lower = text.lower()
            text_token = ViTokenizer.tokenize(text_lower)
            data.append(text_token)
        file.close()
    stop_words= []
    with open("./Stopword/vietnamese-stopwords.txt", encoding="utf-8") as f:
        text = f.read()
        for word in text.split():
            stop_words.append(word)
        f.close()
    punc_ = list(punctuation)
    stop_word = stop_words + punc_
    sentences = []
    for d in data:
        sent = []
        for word in d.split(" "):
            if word not in stop_word:
                if "_" in word or word.isalpha() is True:
                    sent.append(word)
        sentences.append(" ".join(sent))
    return sentences
    pass


def write_data_clean(data_clean, document):
    path = './DataClean' + '/' + document + '_clean/'
    try:
        os.makedirs('DataClean' + '/' + document + '_clean')
    except FileExistsError:
        pass
    for text in range(len(data_clean)):
        num_file = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
        f = open(path + str(num_file) + "_clean.txt", "w+", encoding='utf-8')
        f.write(data_clean[text] + "\n")
        f.close()
    pass


def vectorizer(in_):
    vect = TfidfVectorizer(use_idf=True)
    x_ = vect.fit_transform(in_)
    return x_


def raw_data():
    for i in range(len(menu_list)):
        Document = menu_list[i].get('title')
        if Document in ['Thi viết về GTVT', 'Video thời sự', 'Giải trí - thể thao']:
            continue
        item_url = requests.get(menu_list[i].get('href'))
        item_soup = get_soup(item_url)
        articles_link = get_articles(item_soup)
        for j in range(len(articles_link)):
            """Lấy tiêu đề, phụ đề, ngày đăng và nội dung của bài báo lưu về file"""
            Title, Subtitle, Date, Content = get_data(articles_link[j])
            write_data_raw(Title, Subtitle, Date, Content, Document)
        """ Mở tệp trong DataRaw và lọc các stopword, dấu câu không cần thiết"""
        data_cleaned = load_and_clean_data(Document)
        write_data_clean(data_cleaned, Document)
        assign_label_data(Document)
    pass


def get_label(doc):
    labels = []
    paths = glob.glob("./DataLabel/" + doc + "/*.txt")
    for path in paths:
        with open(path, encoding="utf-8") as file:
            text = file.read()
            labels.append(text)
        file.close()
    return labels
    pass


def get_cleaned(doc):
    X_ = []
    paths = glob.glob("./DataClean/" + doc + "_clean/*.txt")
    for path in paths:
        with open(path, encoding="utf-8") as file:
            text = file.read()
            X_.append(text)
        file.close()
    return X_
    pass


def create_data_train():
    labels_ = []
    X = []
    for i in range(len(menu_list)):
        Doc = menu_list[i].get('title')
        if Doc in ['Thi viết về GTVT', 'Video thời sự', 'Giải trí - thể thao']:
            continue
        labels_ += get_label(Doc)
        X += get_cleaned(Doc)
    with open('data_train.csv', mode='w', encoding='utf-8') as training_file:
        writer = csv.writer(training_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(labels_)):
            writer.writerow([X[i], labels_[i]])


def result(user_input_text):
    feature = ['mess', 'emotion']
    dataset = pd.read_csv('./data_train.csv', header=None, names=feature)
    dataset['emotion_num'] = dataset.emotion.map({'Negative': 0, 'Positive': 1})
    X = dataset.mess
    y = dataset.emotion_num
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    vect = TfidfVectorizer()
    X_ = vect.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_, y)
    x_text = clean_dat_text(user_input_text)
    test = vect.transform(x_text)
    positive_prob = float(model.predict_proba(test)[:, 1])
    negative_prob = 1 - positive_prob
    """ Tính độ chính xác"""
    X_test_dtm = vect.transform(X_test)
    y_pred_class = model.predict(X_test_dtm)
    accuracy = metrics.accuracy_score(y_test, y_pred_class)
    return positive_prob, negative_prob, accuracy


if __name__ == '__main__':
    main_url = requests.get('https://www.baogiaothong.vn')
    soup = get_soup(main_url)
    title_passed = []
    menu_list = []  # """ Lưu lại link và title menu_item """
    for ul in soup.find_all('ul', class_='sub-menu', id=["menu_child_2", "menu_child_69"]):
        for li in ul.find_all('li'):
            href = li.find('a')
            menu_list.append({'href': href['href'], 'title': href['title']})
    # raw_data()
    # create_data_train()
    prob_pos, prob_neg, acc = result("Tai nạn giao thông ngày càng nhiều")
    print(prob_neg)
    print(prob_pos)
    print(acc)
