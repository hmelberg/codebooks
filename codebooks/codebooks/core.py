# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 02:35:26 2019

@author: hmelberg_adm
"""

import pandas as pd
import pickle
import re
import glob
from dataclasses import dataclass
from typing import Any

BOOKS_PATH = './books/'


# %%
def load(file):
    """
    Reads a codebook into memory.

    Args:
        file (str): filename (to a pickle object)

    Note:
        codebook objects are stored using the pickle format

    """
    with open(file, 'rb') as input:
        rb = pickle.load(input)
    return rb


def _search_text(df, text, cols='text', method='logical expression'):
    """
    Searches column(s) in a dataframe for occurrences of words or phrases

    Can be used to search for occurrences codes that are associated with certain words in the text description.

    Args:
        df (dataframe or series) : The dataframe or series with columns to be searched
        cols (str or list of str): The columns to be searched.
        raw (bool): if True, searches for the raw textstring without modifications
        regex (bool): If True, use regex when searching
        logic (bool): If True, use logical operators in the search (and, or, not in the string)

    Returns:
        A dataframe with the rows that satisfy the search conditions (contain/not contain the words/phrases the user specified)
        Often: The codes where the description contain certain words

    Examples:
        icd.search_text('diabetes')
        icd.search_text('diabetes and heart')
        icd.search_text('cancer and not (breast or prostate)')

    """
    #     Strcture
    #        0. Identify phrases and substitute space in phrases with underscores to the phrase is considered to be one word
    #        1. find all whole words
    #        2. select search methode depending on input (search for raw text, search using regex, search using logical operators etc)
    #        3. replace all terms with '{term}==1, but not replace ogical operator terms
    #        4. create str.contains bool col for every term
    #        5. run pd eval

    if isinstance(cols, str):
        cols = [cols]

    ## find all whole words used in the text

    # first: words within quotation marks (within the string) are to be considered "one word"
    # to make this happen, replace space in text within strings with underscores
    # then the regex will consider it one word - and we reintroduce spaces in texts with stuff with underscore when searching
    if not raw:
        phrases = re.findall(r'\"(.+?)\"', text)
        for phrase in phrases:
            text = text.replace(phrase, phrase.replace(' ', '_'))

    # find all words
    word_pattern = r'\w+'
    words = set(re.findall(word_pattern, text))
    skipwords = {'and', 'or ', 'not'}
    if not raw:
        words = words - skipwords
    rows_all_cols = len(df) * [False]  # nb common mistake

    # conduct search: either just the raw text, the regex, or the one with logical operators (and, or not)
    if method == 'raw':
        for col in cols:
            rows_with_word = df[col].str_contains(text, na=False, regex=False)
            rows_all_cols = rows_all_cols | rows_with_word  # doublecheck!
    elif method == 'pure regex':
        for col in cols:
            rows_with_word = df[col].str_contains(text, na=False, regex=True)
            rows_all_cols = rows_all_cols | rows_with_word  # doublecheck!
    elif method == 'logical expression':
        for col in cols:
            for word in words:
                name = word
                # words with underscores are phrases and underscores must be removed before searching
                if ('_' in word) and (has_underscore): word = word.replace('_', ' ')
                df[name] = df[col].str.contains(word, na=False)
            all_words = re.sub(r'(\w+)', r'\1==1', text)
            # inelegant, but works
            for word in skipwords:
                all_words = all_words.replace(f'{word}==1', word)
            rows_with_word = df.eval(all_words)  # does the return include index?
        rows_all_cols = rows_all_cols | rows_with_word  # doublecheck!
    else:
        for col in cols:
            rows_with_word = df[col].str_contains(text, na=False, regex=False)
            rows_all_cols = rows_all_cols | rows_with_word  # doublecheck!

    return rows_all_cols


# %%
from dataclasses import dataclass


@dataclass
class Data(datadikt == None):
    """
    Class containing meta information about a codebook dataframe.'''
    """
    code_col: str = 'code'
    label_col: str = 'label'
    description: str = None
    keywords: set = None
    download_url: str = None
    links: list = None

    parent: str = None
    child: list = None
    sibling: list = None


from types import SimpleNamespace


# %%
class Book:
    standard = {'name': None,
                'code_col': 'code',
                'label_col': 'label',
                'description_col': None,

                'description': None,
                'keywords': None,
                'url': None,
                'links': None,

                'parent': None,
                'child': None,
                'sibling': None}

    def __init__(self, name, df=None, **kwargs):
        # maybe chapter is better do not instert books into books, but can have chapters inside chapters!
        self.df = df
        data = Book.standard.update(**kwargs)
        self.data = SimpleNamespaces(**data)

    @property
    def values(self):
        return self.df[self.code].tolist()

    def insert(self, name, df=None, **kwargs):
        """
        Inserts a new codebook or codelist

        Args:
            name: Name of the new codebook or codelist
            content (dataframe, series, list): the dataframe, series or list representing the codebook
                If it is a list or a series, it wil be converted to a dataframe
            code: if it is a dataframe, the name of the column with the codes
            label: if it is a dataframe, the name of the column with the labels
            data: A dictionary with metainformation about the codebook inserted (keywords, description, url, cols etc)
        """

        if df is None:
            df = globals()[name]
        if isinstance(df, list):
            category = 'codelist'
            content = pd.DataFrame(df, columns=['code'])
        if not keywords:
            keywords = {name}
        elif isinstance(keywords, str):
            keywords = {keywords.split(',')}

        data = Book.standard.update(**kwargs)
        data['name'] = name
        data['parent'] = self.data.name

        book = Book(name, df=df, **data)
        setattr(self, name, book)

    def add(self, codes):
        """
        Adds information, like codes, to the codebook

        codes:
            if str, adds the singel code to the codebook (no label)
            if list of str, adds all the codes in the list to the codebook (no label)
            if dict, adds the key as the code and value as the label to the codebook
            if a dataframe, appends the dataframe to the existing codebook
        """
        if isinstance(codes, dict):
            row = [self.data.code:codes.key(), self.data.label = codes.value()]
            elif isinstance(codes, str):
            row = {self.data.code: codes}
        elif isinstance(codes, list) & isinstance(code[0], str):
            row = [{self.data.code: code} for code in codes]
        elif isinstance(codes, list) & isinstance(code[0], dict):
            row = [[{self.data.code: code.key(), self.data.label: code.value()}] for code in codes]  # hmm test this

        elif isinstance(codes, pd.DataFrame):
            row = codes
            # also allow series?
        # error if not list, dict or str used as input, maybe allow a df?
        self.df = self.df.append(row, ignore_index=True)  # axis?

    def search(self, text, cols=None, raw=False, regex=False, logic=True, has_underscore=False, out='code_and_label'):
        """
        Searches a columns in the codeboko for words, logical operators and regex is allowed

        Args:
            text (str): text or logical expression, or regex to be used in the search
            cols (str, list of str): the columns in which to search, default is the column with the code labels
            raw (bool): if true, searches for the raw text, not reges, notation or logical operators are used
            regex (bool): if True, allows use of regex in the search
            logic (bool): if True, allows use of logical operators and, or , not in the search
            has_underscores (bool): # to be depreciated
            out (str):
                if 'code_and_label' returns a dataframe with those
                if 'codes', returns a list of only the codes that satisfy the search criteria
                if 'df', returns all the columns for the rows that satisfy the search criterion

        """
        if not cols:
            cols = self.data.label
        select = search_text(df=self.df, text=text, cols=cols, raw=raw, regex=regex, logic=logic,
                             has_underscore=has_underscore)
        if out == 'code_and_label':
            return self.df[select][[self.data.code, self.data.label]]
        elif out == 'codes':
            return self.df[select][self.data.code].tolist()
        elif out == 'df':
            return self.df[select]
        else:
            print(f'Error: out argument {out} is not unsupported')
            return None

    def lookup(self, codes=None, case=True, regex=True, flags=0, notation=True, out='code_and_label'):
        """
        Looks up code(s) in the codebook and returns informaiton about the code(s)

        Args:
            codes str or list of str: the code(s) to look for (allow notation, regex)
            case (bool): If True (default), the lookup is case sensitive
            regex (bool): if True, allow the lookupstring to be a regex
            raw (bool): If True, searchs for the raw string, no regex, notation
            out (str):
                if 'code_and_label' returns a dataframe with those
                if 'codes', returns a list of only the codes that satisfy the search criteria
                if 'df', returns all the columns for the rows that satisfy the search criterion
        """
        if isinstance(codes, str):
            codes = [codes]
        expanded = []
        for code in codes:
            expanded_one = self._expand_one(code=code, case=case, regex=regex, flags=flags, notation=notation)
            expanded.extend(expanded_one)

        selection = self.df[self.data.code].isin(expanded)
        if out == 'code_and_label':
            return self.df[selection][[self.data.code, self.data.label]]
        elif out == 'df':
            return self.df[selection]
        elif out == 'codes':
            self.df[selection][self.data.code].tolist()
        else:
            print(f'Error: Out argument {out} is not valid. Allowed options are: code_and_lable, df, codes')
            expanded = ', '.join(expanded)
        return expanded

    def _expand_one(self, code, case=True, regex=True, flags=0, notation=True):
        """
        Takes a given code expression and return all codes that match it

        Example 'S72*' will return all codes starting with S72

        """
        if notation & regex:
            # make regex expression
            if '*' in code:
                code = '^' + code.replace('*', '.*') + '$'
            elif ':' in code:
                # maybe sort after order (if order column exist)
                code = self.df.set_index(self.code)[code].tolist()
                return code
            # do not implement '-' notation? Useless (since have :) and ambigous
        selection = self.df[self.data.code].str.contains(code, case=case,
                                                         regex=regex, flags=flags)
        expanded = self.df[self.data.code][selection].tolist()
        return expanded

    def save(self, file, protocol=4):
        """
        saved the (specific) codebook object (pickleformat)

        Note: Saves all objects in and below the given level, not books above in the codebook hierarchy
        """
        with open(file, 'wb') as output:
            pickle.dump(self, output, protocol=protocol)

    def view(self, what):
        """
        Shows different types of information about the codebook
        """
        if what == 'codes':
            print(self.values)
        elif what == 'children':
            print
            self.data['children']
        elif what == 'parent':
            print
            self.data['parent']
        # elif what == 'closest headline':
        # elif what == 'group_name', level=0, 1, 2, etc i.e. closest group name (that itself is not/may not be a valid code, up or down, or at a given level)
        # elif? keywords etc?
        # idea, but not here: code relatives (ex. heart codes in hospital and heart codes in atc? orprimary care?)
        # idea: all codes relating to a given part of the body
        # all codes that are disallowed with other codes
        # code rules

    def split(self, groupby):
        """
        Splits a codebook into seperate and smaller codebooks

        Args:
            groupby (str): The column that determines how to split the codebook

        Note:
            Codebooks in the hierarchy below are kept intact, under the old name
            The old dataframe is deleted since the information is now in the new and separate books
        # todo: is a merge command also useful here?
        """
        groups = self.df.groupby(groupby)
        code = self.code
        label = self.label

        for name, group in groupby:
            self.insert(name, df=group)
        del self.df


# %%
path = "c://dat/us_synth/"
ndc = pd.read_csv(path + 'ndc_us_2012_from_nber_rx.csv', sep=',', encoding='latin-1')
ndc.head()
ndc.columns
# ndc[ndc.npropname.str.contains('fluoxetine', case=False, na=False)]
icd = pd.read_csv(path + 'icd9_dx_2015_from_nber.csv', sep=',', encoding='latin-1')

icd.set_index('dgns_cd')['0019':'0021'].index.tolist()
icd.columns

s = icd.dgns_cd.str.startswith('72')
e = icd.dgns_cd.str.endswith('5')
(s & e).sum()

icd.dgns_cd[icd.dgns_cd.str.contains('^72.*5$')]

cb = Codebooks()
cb.add('icd', code='dgns_cd', label='shortdesc')
cb.view()
cb
cb.icd
cb.icd.df
cb.icd.add('heart', ['s1', 's2'])
cb.icd.heart.values
cb.icd.heart.append(['s5', 's7'])
cb.icd.heart.df
cb.icd.heart.df = cb.icd.heart.df.append({'code': 's8'}, ignore_index=True)

cb.icd.heart.codes()
book.icd.heart.codes()

book.icd.heart.insert(name, code=, label=, data=...)
book.icd.heart.update()
book.icd.heart.change('code.str.upper()')  # may have recording/history of change also

codes.icd.heart.insert_code('s72')

book.icd.heart.codes
better?
codes.icd.heart.values

df.first
codes['icd']['heart']
[]

# have dedfault books, for label

codes.icd.heart
codes.atc.heart
codes.icd.add()

codes.icd.heart.search()

book.icd.heart.add_codelist(...)..no?




cb.icd.heart = ['s73', 's73']

cb.icd.expand('V90*, V91*', out='str')
cb.icd.explain('V90*')
cb.icd
cb.icd.codes[-20:]

cb.icd.df.dgns_cd = cb.icd.df.dgns_cd.str[0:2]

del icd

codes.icd.list

cb.icd.search('fracture')

cb.icd.df
codes.icd.all
list
show
view
codes.icd.codes
codes.icd.values

cb.icd.search('fracture and not traumatic')
cb.icd.search('malign and NEC')
cb.icd.search()

.sum()

icd.columns
icd.dgns_cd.str.contains('^V91.*$')
icd[icd.shortdesc.str.contains('hip')]

https: // clinicalcodes.rss.mhs.man.ac.uk / medcodes / articles /
https: // clinicalcodes.rss.mhs.man.ac.uk / medcodes / article / 14 / ro /
https: // clinicalcodes.rss.mhs.man.ac.uk / medcodes / article / 14 / codelist / res14 - upper - gastrointestinal - malignancy / download /


# %%
class Codebooks:
    def __init__(self):
        self.books = []
        self.booklist = []
        self.description = {}
        self.path = None

    def add(self, name, df=None, keywords=None, code='code', label='label', cols=None, comments=None, meta=None,
            etc=None, download_url=None):

        if not df:
            df = globals()[name]
        if not keywords:
            keywords = {name}
        elif isinstance(keywords, str):
            keywords = {keywords.split(',')}

        # self.description[name]=description
        book = Book(name, df=df, keywords=keywords, code=code, label=label, cols=cols, comments=comments, meta=meta,
                    etc=etc)
        setattr(self, name, book)
        self.books.append(book)
        self.booklist.append(name)

    def add_from_path(self, path=BOOKS_PATH, contains=None, sep='_'):
        files = glob(path)
        for file in files:
            words = file.split('/')[-1].split(sep)
            name = word[0]
            keywords = words
            if contains.issubset(words):
                df = pd.read_csv(file)
                if name in self.booklist:
                    x = 1
                    while name in booklist:
                        name = name + f'_{x}'
                        x += 1
                if 'code' not in df.columns | 'text' not in df.columns:
                    print("Error: The dataframe does not have columns named 'code' and 'label'")
                    continue

                self.add(name=name, df=df, keywords=keywords, code='code', label='text')

    def add_from_web(self, url):
        pass

    def delete(self, name):
        if isinstance(name, str):
            name = [name]
        for nam in name:
            self.books.remove(nam)
            delattr(self, nam)

    def save(self, name, protocol=4):
        with open(file, 'wb') as output:
            pickle.dump(self, output, protocol=protocol)

    def view(self, name=None):
        if not name:
            name = self.books
        elif isinstance(name, str):
            name = [name]
        for book in name:
            print(book.name, book.keywords)


# %%


cb = Codebooks()
cb.add('ndc')

cb.view()

cb.ndc.df
cb.ndc.keywords.add('2010')


# %%

def search(self, terms, book=None, cols=None, split=True):
    pass
    # maybe get_rows is better, or maybe a different method that should be added


def expand(self):
    rb.atc.expand('4AB*')  # should return full codelist (optinal labels also?)
    # but then the method has to be on the book!
    # alternative: rb.expand('4AB*', 'atc')


def code(self):
    rb.atc.code('4AB02')  # should return? label? row? yes, label

    rb.atc.expand('4AB*')  # should return full codelist (optinal labels also?)


def get_codes(self):


def find_codes(self):


def extract_codes(self):


def list_codes


cb = Codebooks()
cb.add('ndc', ndc)

isinstance(globals()['ndc'], pd.DataFrame)

ndc.name

cb.ndc
ndc = ndc.sample(10000)

cb.ndc = cb.ndc.sample(10000)

url = 'https://clinicalcodes.rss.mhs.man.ac.uk/medcodes/articles/'

df = pd.read_html(url)[0]
df
df.columns
df = df[0]

df.Title

https: // clinicalcodes.rss.mhs.man.ac.uk / medcodes / article / 1 /

aurl = 'https://clinicalcodes.rss.mhs.man.ac.uk/medcodes/article/1/'

cbooks = pd.read_html(aurl)[0]

c = 'https://clinicalcodes.rss.mhs.man.ac.uk/medcodes/article/1/codelist/chd/'

codes = pd.read_html(c)[0]
codes.columns
https: // clinicalcodes.rss.mhs.man.ac.uk / medcodes / article / 1 / codelist / chd / download /

# author also?

articles_url = 'https://clinicalcodes.rss.mhs.man.ac.uk/medcodes/articles/'

articles = pd.read_html(articles_url)[0]
source = {}
all_codes = {}
articles = articles[5:8]

# for n, arow in enumerate(articles.itertuples(), 1):
for n in range(1, 71):
    # comment=f"Title: {arow.Title}, Journal: {arow.Journal}, Authors: {arow.Authors}, Year: {arow.Year}"
    aurl = f'https://clinicalcodes.rss.mhs.man.ac.uk/medcodes/article/{n}/'
    try:
        codelists = pd.read_html(aurl)[0]['Code list'].tolist()
    except:
        print('Error, article', aurl)
        continue
    for codelist in codelists:
        codename = codelist.replace(': ', '-').replace(' - ', '-').replace(' ', '-').replace('/', '').replace('&', '')
        url = f'https://clinicalcodes.rss.mhs.man.ac.uk/medcodes/article/{n}/codelist/{codename.lower()}/'
        try:
            codes = pd.read_html(url)[0]
            name = codename.lower() + '_' + codes['Coding system'][0] + '_' + f'{n}'
            source[name] = url
            codes['source'] = url
            all_codes[name] = codes
        except:
            print('Error, codelist', url)

df = pd.DataFrame(list(all_codes.values()))
df = pd.concat(list(all_codes.values()), axis=0, ignore_index=True)
df.columns
df.head()
df[['Code', 'Description', 'List name', 'source']]

df.to_csv('clinicalcodes_codes_v1.csv')
articles.to_csv('clinicalcodes_articles_v1.csv')


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('name + '.pkl', 'rb') as f:
    return pickle.load(f)


save_obj(all_codes, 'dict_of_df_clin_codesv1.p')

df['source'].iloc[-1]
df.iloc[-1]

len(df)
all_codes.values()

all_c_copy = all_codes.copy()

url = f'https://clinicalcodes.rss.mhs.man.ac.uk/medcodes/article/{n}/'
cbooks = pd.read_html(aurl)[0]

comments = f"Article: {Title}, Journal
cbooks = pd.read_html(aurl)[0]

c = 'https://clinicalcodes.rss.mhs.man.ac.uk/medcodes/article/1/codelist/chd/'

codes = pd.read_html(c)







