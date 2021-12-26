## Imports
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



## Wrangle function
def wrangle(filepath):
    df = pd.read_csv(filepath)
    
    # column names
    df.columns = df.columns.str.strip().str.lower().str.replace('.','_')
    df['argdomainratio'] = df['argdomanratio']
    df.drop(columns=['argdomanratio'], inplace=True)
    
    # Drop constant columns
    [df.drop(columns=col, inplace=True) for col in df.columns if df[col].nunique() == 1]
    
    # Replacing the ten infinite values with nulls 
    df.replace([np.inf,-np.inf], value=None, inplace=True)
    
    # Making the target vector lowercase
    df['url_type_obf_type'] = df['url_type_obf_type'].apply(str.lower)
    
    return df

        
## Function for character continuity rate
#### Malicious URL Filtering – A Big Data Application
#### Min-Sheng Lin, Chien-Yi Chiu, Yuh-Jye Lee and Hsing-Kuo Pao
#### Dept. of Computer Science and Information Engineering
#### National Taiwan Univ. of Science and Technology
def char_continuity(url_string):
    longest_numeric_token = 0
    longest_alphabet_token = 0
    longest_symbol_token = 0
    n = 0
    numeric_list = ['0','1','2','3','4','5','6','7','8','9']
    alphabet_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    for i,character in enumerate(url_string):
        if n > i:
            continue
        if character in numeric_list:
            temp_character = character
            numeric_token_length = 0
            n = i
            while temp_character in numeric_list:
                numeric_token_length += 1
                n += 1
                if n > (len(url_string) - 1):
                    break
                temp_character = url_string[n]
            if numeric_token_length > longest_numeric_token:
                longest_numeric_token = numeric_token_length
        elif character in alphabet_list:
            temp_character = character
            alphabet_token_length = 0
            n = i
            while temp_character in alphabet_list:
                alphabet_token_length += 1
                n += 1
                if n > (len(url_string) - 1):
                    break
                temp_character = url_string[n]
            if alphabet_token_length > longest_alphabet_token:
                longest_alphabet_token = alphabet_token_length
        else:
            temp_character = character
            symbol_token_length = 0
            n = i
            while (temp_character not in numeric_list) & (temp_character not in alphabet_list):
                symbol_token_length += 1
                n += 1
                if n > (len(url_string) - 1):
                    break
                temp_character = url_string[n]
            if symbol_token_length > longest_symbol_token:
                longest_symbol_token = symbol_token_length                
            
    charcontinuityrate = (longest_alphabet_token + longest_numeric_token + longest_symbol_token) / len(url_string)
    return charcontinuityrate

## Function for finding the length of the longest word
def longest_word(component):
    alphabet_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    n = 0
    longest_word_length = 0
    for i,character in enumerate(component):
        if n > i:
            continue
        if character in alphabet_list:
            temp_character = character
            word_length = 0
            n = i
            while temp_character in alphabet_list:
                word_length += 1
                n += 1
                if n > (len(component) - 1):
                    break
                temp_character = component[n]
            if word_length > longest_word_length:
                longest_word_length = word_length
    return longest_word_length

## Function for calculating ratio of url components
## Used for pathurlratio, argurlratio, domainurlratio, pathdomainratio, argpathratio
def url_ratio(string_1, string_2):
    if len(string_2) != 0:
        ratio = len(string_1) / len(string_2)
    else:
        ratio = 0
    return ratio

## Function for calculating LDL and DLD
## Returns (LDL,DLD)
def letter_digit(url_string):
    ldl_count = 0
    dld_count = 0
    length_url_string = len(url_string)
    number_of_checks = length_url_string-2
    numeric_list = ['0','1','2','3','4','5','6','7','8','9']
    alphabet_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    for character in range(number_of_checks):
        temp_string = url_string[character:character+3]
        if temp_string[0] in numeric_list and temp_string[1] in alphabet_list and temp_string[2] in numeric_list:
            dld_count+=1
        elif temp_string[0] in alphabet_list and temp_string[1] in numeric_list and temp_string[2] in alphabet_list:
            ldl_count+=1
        else:
            None
    return ldl_count, dld_count 

def num_let_sym_count(component):
    numcount = 0
    letcount = 0
    symcount = 0
    numeric_list = ['0','1','2','3','4','5','6','7','8','9']
    alphabet_list = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    for character in component:
        if character in numeric_list:
            numcount += 1
        elif character in alphabet_list:
            letcount += 1
        else:
            symcount += 1
    
    return [numcount, letcount, symcount]

def token_len(item_list):
    total=0
    longtokenlen = 0
    for item in item_list:
        total = total+len(item)
        if len(item) > longtokenlen:
            longtokenlen = len(item)
    avgtokenlen = total / len(item_list)
    return avgtokenlen, longtokenlen

def dots_in_url(url):
    count=0
    for n in url:
        if n == '.':
            count = count+1
    return count

def number_rate(component):
    numcount = 0
    numeric_list = ['0','1','2','3','4','5','6','7','8','9']
    for character in component:
        if character in numeric_list:
            numcount += 1
    if len(component) != 0:
        numberrate = numcount / len(component)
    else:
        numberrate = -1
    return numberrate

def url_parse(url):
    url = url.replace('https://','').replace('http://','')
    urllen = len(url)
    path_exists = False
    filename = ''
    file_exists = False
    query_exists = False
    path_token_count = 0
    avgpathtokenlen = -1
    longestpathtokenlen = 0
    ldl_path = 0
    ldl_filename = 0
    ldl_getarg = 0
    dld_filename = 0
    dld_path = 0
    dld_getarg = 0
    subdirlen = 0
    pathurlratio = 0
    argdomainratio = 0
    argurlratio = 0
    pathdomainratio = 0
    argpathratio = 0
    urlqueries_variable = -1
    pathlength = -1
    longestvariablevalue = -1
    
    domain = url
    if '/' in url:
        domain, not_domain = url.split('/',maxsplit=1)[0], url.split('/',maxsplit=1)[1]
        domainlength = len(domain)
        longestvariablevalue = -1
        if len(not_domain) != 0:
            path_exists = True
        if '?' in not_domain:
            path, query = not_domain.split('?', maxsplit=1)[0], not_domain.split('?',maxsplit=1)[1]
            query_list = query.split('&')
            querylength = len(query)
            urlqueries_variable = len(query_list)
            query_exists = True
            for variable in query_list:
                if len(variable) > longestvariablevalue:
                    longestvariablevalue = len(variable)       
        else:
            querylength=0
            path = not_domain
            query = ''
        pathlength = len(path)
        path_list = path.split('/')
        if len(path_list[-1]) != 0:
            filename = path_list[-1]
            file_exists = True
        if '.' in filename:
            this_fileextlen = len(filename.split('.')[-1])
            file_extension = filename.split('.')[-1]
            filename = filename.split('.')[0]
        else:
            file_extension = 'html'
            this_fileextlen = 4
    filenamelen = len(filename)
    domain_list = domain.split('.')
    tld_class = domain.split('.')[-1]
    domain_token_count = len(domain_list)
    if path_exists:
        path_token_count = len(path_list)
    avgdomaintokenlen, longdomaintokenlen = token_len(domain_list)
    if path_exists:
        avgpathtokenlen, longestpathtokenlen = token_len(path_list)
        subdirlen = pathlength
    else:
        querylength = 0
        file_extension = 'html'
        domainlength = len(domain)
        this_fileextlen = len(file_extension)
    ## Calling num_let_sym_count(component)
    ## for various features.
    ## Some features are embedded in 
    ## loops to be sure they exist.
    url_digitcount, url_letter_count, symbolcount_url = num_let_sym_count(url)
    host_digitcount, host_letter_count, symbolcount_domain = num_let_sym_count(domain)
    if path_exists:
        directory_digitcount, directory_lettercount, symbolcount_directoryname = num_let_sym_count(path)
    else:
        directory_digitcount, directory_lettercount, symbolcount_directoryname = [-1,-1,-1]
    if file_exists:
        file_name_digitcount, filename_lettercount, symbolcount_filename = num_let_sym_count(filename)
    else:
        file_name_digitcount, filename_lettercount, symbolcount_filename = [-1,-1,-1]
    if query_exists:
        query_digitcount, query_lettercount, symbolcount_afterpath = num_let_sym_count(query)
    else:
        query_digitcount, query_lettercount, symbolcount_afterpath = [-1,-1,-1]
    extension_digitcount, extension_lettercount, symbolcount_extension = num_let_sym_count(file_extension)
    ## Calling dots_in_url(url) to count
    ## occurence of periods
    numberofdotsinurl = dots_in_url(url)
    
    ## Calling char_continuity(url) for
    ## the character continuity rate
    charcontinuityrate = char_continuity(url)
    
    ## Calling url_ratio(string_1,string_2) for
    ## pathurlratio, argurlratio, domainurlratio, pathdomainratio, argpathratio
    if path_exists:
        pathurlratio = url_ratio(path,url)
        pathdomainratio = url_ratio(path,domain)
    domainurlratio = url_ratio(domain,url)
    if query_exists:
        argurlratio = url_ratio(query,url)
        argdomainratio = url_ratio(query,domain)
    if query_exists and path_exists:
        argpathratio = url_ratio(query,path)
    
    ## Calling letter_digit() for ldl and dld
    ldl_url, dld_url = letter_digit(url)
    ldl_domain, dld_domain = letter_digit(domain)
    if path_exists:
        ldl_path, dld_path = letter_digit(path)
    if file_exists:
        ldl_filename, dld_filename = letter_digit(filename)
    if query_exists:
        ldl_getarg, dld_getarg = letter_digit(query)
    
    ## Calling number_rate(component) 
    numberrate_url = number_rate(url)
    numberrate_domain = number_rate(domain)
    numberrate_extension = number_rate(file_extension)
    if path_exists:
        numberrate_directoryname = number_rate(path)
    else:
        numberrate_directoryname = -1
    if file_exists:
        numberrate_filename = number_rate(filename)
    else:
        numberrate_filename = -1
    if query_exists:
        numberrate_afterpath = number_rate(query)
    else:
        numberrate_afterpath = -1
        
    ## Calling longest_word(component)
    domain_longestwordlength = longest_word(domain)
    if path_exists:
        path_longestwordlength = longest_word(path)
    else:
        path_longestwordlength = -1
    if file_exists:
        subdirectory_longestwordlength = longest_word(filename)
    else:
        subdirectory_longestwordlength = -1
    if query_exists:
        arguments_longestwordlength = longest_word(query)
    else:
        arguments_longestwordlength = -1
        
    data =  {'querylength':querylength,
            'domain_token_count':domain_token_count, 
            'path_token_count':path_token_count, 
            'avgdomaintokenlen':avgdomaintokenlen, 
            'longdomaintokenlen':longdomaintokenlen, 
            'avgpathtokenlen':avgpathtokenlen,
            'ldl_url':ldl_url,
            'ldl_domain':ldl_domain,
            'ldl_path':ldl_path,
            'ldl_filename':ldl_filename,
            'ldl_getarg':ldl_getarg,
            'dld_url':dld_url,
            'dld_domain':dld_domain,
            'dld_path':dld_path,
            'dld_filename':dld_filename,
            'dld_getarg':dld_getarg, 
            'urllen':urllen, 
            'domainlength':domainlength, 
            'pathlength':pathlength,
            'subdirlen':subdirlen, 
            'filenamelen':filenamelen, 
            'this_fileextlen':this_fileextlen,
            'pathurlratio':pathurlratio,
            'argurlratio':argurlratio,  
            'domainurlratio':domainurlratio, 
            'pathdomainratio':pathdomainratio,
            'argpathratio':argpathratio,
            'numberofdotsinurl':numberofdotsinurl,
            'charcontinuityrate':charcontinuityrate,
            'longestvariablevalue':longestvariablevalue,
            'url_digitcount':url_digitcount,
            'host_digitcount':host_digitcount,
            'directory_digitcount':directory_digitcount,
            'file_name_digitcount':file_name_digitcount,
            'extension_digitcount':extension_digitcount,
            'query_digitcount':query_digitcount,
            'url_letter_count':url_letter_count,
            'host_letter_count':host_letter_count,
            'directory_lettercount':directory_lettercount,
            'filename_lettercount':filename_lettercount,
            'extension_lettercount':extension_lettercount,
            'query_lettercount':query_lettercount,
            'longestpathtokenlen':longestpathtokenlen,
            'domain_longestwordlength':domain_longestwordlength,
            'path_longestwordlength':path_longestwordlength,
            'sub-directory_longestwordlength':subdirectory_longestwordlength,
            'arguments_longestwordlength':arguments_longestwordlength,
            'urlqueries_variable':urlqueries_variable,
            'numberrate_url':numberrate_url,
            'numberrate_domain':numberrate_domain,
            'numberrate_directoryname':numberrate_directoryname,
            'numberrate_filename':numberrate_filename,
            'numberrate_extension':numberrate_extension,
            'numberrate_afterpath':numberrate_afterpath,
            'symbolcount_url':symbolcount_url,
            'symbolcount_domain':symbolcount_domain,
            'symbolcount_directoryname':symbolcount_directoryname,
            'symbolcount_filename':symbolcount_filename,
            'symbolcount_extension':symbolcount_extension,
            'symbolcount_afterpath':symbolcount_afterpath,
            'argdomainratio':argdomainratio
            }
    df = pd.DataFrame(data=data, index=[0])
    return df