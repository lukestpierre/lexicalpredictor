import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import functions
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


df = pd.read_csv('fulldata.csv')

df.drop(columns=['Unnamed: 0'], inplace=True)

X = df.drop(columns=['label'])
y = df['label']

X_train, X_val, y_train, y_val = train_test_split(X,
                                                  y,
                                                  random_state=42,
                                                  train_size=0.9)

model_rf = make_pipeline(
    SimpleImputer(strategy='constant',
                  fill_value=-1),
    RandomForestClassifier(random_state=42,
                           max_depth=32)
)

model_gb = make_pipeline(
    SimpleImputer(strategy='constant',
                  fill_value=-1),
    GradientBoostingClassifier(random_state=42)
)

model_rc = make_pipeline(
    SimpleImputer(strategy='constant',
                  fill_value=-1),
    RidgeClassifier(random_state=42, alpha=4589)
)

model_rf.fit(X_train,y_train)

model_gb.fit(X_train,y_train)

model_rc.fit(X_train,y_train)



app = dash.Dash(__name__, suppress_callback_exceptions=True)

server = app.server

app.layout = html.Div(children=[
    html.Ul(children=[
        html.Li(children=[dcc.Link(children='About', href='about')]),
        html.Li(children=[dcc.Link(children='Feature Analysis', href='featureanalysis')]),
        html.Li(children=[dcc.Link(children='Feature Library', href='featurelibrary')]),
        html.Li(children=[dcc.Link(children='Model Analysis', href='modelanalysis')]),
        html.Li(children=[dcc.Link(children='Make Prediction', href='prediction')]),
        html.Li(children=[dcc.Link(children='References', href='references')])
    ]),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])

about_layout = html.Div(
    children=html.Div(id='body-container', children=dcc.Markdown('''
                          # Identifying Malicious URLs with Lexical Items
                          
                          There are billions of indexed URLs on the internet. Although many URLs are simple web pages, many others could be serving as potential
                          endpoints for unsolicited, malicious activity. Being able to predict whether a URL is malicious or not - in real time - could protect users
                          from this vulnerability.
                          
                          ## What are Lexical Items?
                          
                          Lexical items are components that form the basic syntax of a given language. In this case, the URL was broken into varius components 
                          and various lexical features were derived from each component.
                          
                          Some examples include the 'token count', 'digit count', 'average token length',
                          etc. See the [Feature Library](featurelibrary) for more information.
                          
                          ## Metrics
                          
                          Because the model is dealing with an even distribution of classes, the accuracy could be used as a solid baseline score. However,
                          when dealing with security vulnerabilities, it is extremely important to be sure the model is not sacrificing recall of the malicious class
                          for precision or recall elsewhere. Thus, extra attention will be given to precision and recall.
                          
                          ## Choosing the Model
                          
                          The data was fit into a ridge classifier, a random forest classifier, and a gradient boost classifier model using sci-kit learn.
                          For baseline scoring, all default parameters were used, aside from 'random_state=42' where applicable. 
                          
                          As shown above, the baseline accuracy score for the random forest classifier was the highest, while the logistic regression was the lowest.
                          Looking more closely at the precision and recall, the random forest classifier continues to outperform the other two models in all metrics.
                          Based on baseline metrics alone, the ridge classifier and the random forest classifier will be further explored for this project.
                          
                          ## Tuning the Models
                          
                          The primary hyperparameter of concern within the ridge classifier is the alpha parameter. This parameter is the regularization strength
                          that will reduce the varience of the estimates from the model. The parameter was tuned using GridSearchCV and k-fold cross-validation. 
                          After exploration with GridSearchCV, the alpha was set to 4589. Setting the regularization in this way did produce an overall
                          improvement in the validation accuracy score. However, looking more closely at the model's precision and recall shows that recall 
                          is being sacrificed for a slight increase in precision. This could present as a security risk that is not worth the slight increase in accuracy.
                           
                          The random forest classifier showed much more promise with hyperparameter tuning. The first parameter tuned was the max depth of the trees within
                          the forest. Using GridSearchCV and k-fold cross validation, the best depth was found to be 32. This parameter alone showed
                          a minor increase in the validation accuracy, as well as the precision and the recall of the model.
                          
                          ## Problems with the Data
                          
                          The biggest problem with the model are its inability to handle shortened URLs and the lack of exposure to more specific directories within webpages. 
                          For instance, the 'dld_path' feature contains 92% 0's simply because the path portion of the URL does not exist. This will greatly hinder the model's
                          performance when dealing with URLs with longer than average paths. More exposure to these subdirectories will improve the model in the future.
                          
                          '''))
)

feat_analysis_layout = html.Div(id='body-container',children = [
    html.H2(children='Choose a feature to evaluate: '),
    dcc.Dropdown(id='dropdown_list',
                 options= [{'label':x,'value':x} for x in X.columns],
                 value = 'longdomaintokenlen'),
    dcc.Graph(id="graph")
]
)
@app.callback(Output('graph','figure'),
              [Input('dropdown_list','value')])

def feat_analysis_graph(value):
    data = X[value]
    fig = px.histogram(data)
    return fig

feat_library_layout = html.Div(id='body-container',
    children=dcc.Markdown(id='listid', children='''
                          # Feature Library
                          
                          These functions were applied to various components of the URL to derive 61 unique features. The components include:
                          
                        The URL itself
                          
                        Domain
                          
                        Path
                         
                        Query
                         
                        Filename
                        
                        File extension
                          

                          
                          ## Component Length
                          
                          The length of a given component of the url. For domain, it is everything between the scheme and the first '/'. For path, It is everything between
                          the first '/' and the first '?'. For the query, it is everything following the first '?'.                          
                                     
                          ## Token Count
                          
                          Component is split into smaller components around the occurence of a specified delimeter.
                                                       Each smaller component - called a token - is counted for the token count. 
                                                       
                        Example: '/path/path/file' -> split at '/' -> 'path', 'path', 'file' ->
                        add tokens -> path_token_count = 3
                                                                   
                                                                    
                        ## Longest Token Length
                        
                        Component is split into tokens around the occurence of a specified delimeter.
                                                          The length of the tokens is checked and the largest value is selected
                                                          
                        Example:  'www.example.com' -> split at '.' -> 'www', 'example, 'com' ->
                        ckeck lengths -> 3 , 7 , 3 -> choose longest -> longdomaintokenlen = 7
                                                                     
                        ## Average Token Length
                        
                        Component is split into tokens around the occurence of a specified delimeter.
                                                          The length of all tokens are then added together and divided by the total number
                                                          of tokens for the average token length.
                                                          
                        Example:  '/path/path/file' -> split at '/' -> 'path', 'path', 'file' ->
                        add tokens lengths -> 4 + 4 + 4 = 12 -> divide by total number 
                        of tokens -> 12 / 3 -> avgpathtokenlen = 4
                                                                     
                        ## Letter Digit Letter of a component
                        
                        The component is checked in groups of 3 characters. If the 3 characters are in the format 'letter - digit -letter', the LDL count goes up.
                        
                        Example: www.ex3amp4le.com -> checks in groups of 3 characters -> 'www', 'ww.', 'w.e', '.ex', 'ex3' until reaching the end of the string ->
                        two occurences are found -> 'x3a', 'p4l' -> ldl_component = 2
                        
                        ## Digit Letter Digit of a component
                        
                        The component is checked in groups of 3 characters. If the 3 characters are in the format 'digit - letter - digit', the DLD count goes up.
                        
                        Example: www.ex3a4mp4l6e.com -> checks in groups of 3 characters -> 'www', 'ww.', 'w.e', '.ex', 'ex3' until reaching the end of the string ->
                        two occurences are found -> '3a4', '4l6' -> dld_component = 2
                        
                        ## Ratio of one component to another
                        
                        The columns are named in the format 'component1component2ratio' where the ratio is equal to the length of component1 divided by the length of component2.
                        
                        Example: the pathurlratio of www.ex.com/path is the length of the path divided by the length of the URL or 4/15
                        
                        ## Number of dots in URL
                        
                        The number of times the '.' delimeter appears within the URL.
                        
                        Example: www.example.com -> numberofdotsinurl = 2
                        
                        ## Character continuity rate
                        
                        The URL is split at into strings whenever the type of the digit switches between a letter, a number, or a non-alphanumeric character.
                        These strings are checked for the longest occurence of each type. The character continuity rate is then calculated as
                        '(length of longest letter token +  length of longest number token + length of longest symbol token) / total length of the URL'.
                         
                        Example: www.abc567.com  -> split into letters, symbols, and numbers while remaining in order -> 'www','.','abc','567','.','com' ->
                        longest of each type is taken -> longest_letters = 3, longest_numbers = 3, longest_symbols = 1 -> add together and divide by the length of the whole URL ->
                        (3 + 3 + 1) /  14 = 7/14 = charcontinuityrate = 0.5
                        
                        ## Component digit count
                        
                        The total number of numeric digits within a given component.
                        
                        Example: www.exa3mp2le.com -> component_digitcount = 2
                        
                        ## Component letter count
                        
                        The total number of alphabet characters within a given component.
                        
                        Example: www.exa3mp2le.com -> component_lettercount = 13
                        
                        ## Component symbol count
                        
                        The total number of non-alphanumeric characters within a given component.
                        
                        Example: www.exa3mp2le.com -> symbolcount_component = 2
                        
                        ## Longest word of a component
                        
                        The length of the longest string of alphabet charactrs withing a given component. 
                        
                        Example: www.examp3le.com -> length of longest word = 5
                        
                        ## Number rate of a commponent
                        
                        The ratio of numeric characters to the total number of characters within a component. 
                        
                        Example: www.exa3mp2le.com -> numberrate_component = 2/17
                          ''')
)

model_analysis_layout = html.Div(id='body-container',
    children= [
        html.H2(children='Choose a model to evaluate: '),
        dcc.Dropdown(id='dropdown_list_2',
                    options= [{'label':'Random Forest','value':'model_rf'},
                              {'label':'Gradient Boost','value':'model_gb'},
                              {'label':'Ridge Classifier','value':'model_rc'}],
                     value = 'model_rf'),
        dcc.Graph(id='graph_2'),
        html.Div(id='classification-report')
    ]
)

@app.callback(Output('graph_2','figure'),
              Output('classification-report','children'),
              Input('dropdown_list_2','value'))

def feat_analysis_graph(value):
    if value == 'model_rf':
        matrix = confusion_matrix(y_val, model_rf.predict(X_val))
        recall_ben = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        precision_ben = matrix[0][0] / (matrix[0][0] + matrix [1][0])
        recall_mal = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        precision_mal = matrix[1][1] / (matrix[1][1] + matrix [0][1])
        text = '"benign" prec: ' + str(precision_ben) + ' "benign" rec: ' + str(recall_ben) + ' "malicious" prec: ' + str(precision_mal) + ' "malicious" rec: ' + str(recall_mal)
        fig = px.imshow(matrix)
        return fig, text
    elif value == 'model_gb':
        matrix = confusion_matrix(y_val, model_gb.predict(X_val))
        recall_ben = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        precision_ben = matrix[0][0] / (matrix[0][0] + matrix [1][0])
        recall_mal = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        precision_mal = matrix[1][1] / (matrix[1][1] + matrix [0][1])
        text = '"benign" prec: ' + str(precision_ben) + ' "benign" rec: ' + str(recall_ben) + ' "malicious" prec: ' + str(precision_mal) + ' "malicious" rec: ' + str(recall_mal)
        fig = px.imshow(matrix)
        return fig, text
    elif value == 'model_rc':
        matrix = confusion_matrix(y_val, model_rc.predict(X_val))
        recall_ben = matrix[0][0] / (matrix[0][0] + matrix[0][1])
        precision_ben = matrix[0][0] / (matrix[0][0] + matrix [1][0])
        recall_mal = matrix[1][1] / (matrix[1][1] + matrix[1][0])
        precision_mal = matrix[1][1] / (matrix[1][1] + matrix [0][1])
        text = '"benign" prec: ' + str(precision_ben) + ' "benign" rec: ' + str(recall_ben) + ' "malicious" prec: ' + str(precision_mal) + ' "malicious" rec: ' + str(recall_mal)
        fig = px.imshow(matrix)
        return fig, text

references_layout = html.Div(
    children=html.Div(id='body-container', children=dcc.Markdown('''
                        # References
                      
                        ## Malicious URL Filtering - A Big Data Application
                      
                        M. Lin, C. Chiu, Y. Lee and H. Pao, "Malicious URL filtering — A big data application," 2013 IEEE International Conference on Big Data, 2013, pp. 589-596, doi: 10.1109/BigData.2013.6691627.
                        
                        ## Detection of Malicious URLs using Machine Learning Techniques
                        
                        F. Vanhoenshoven, G. Nápoles, R. Falcon, K. Vanhoof and M. Köppen, "Detecting malicious URLs using machine learning techniques," 2016 IEEE Symposium Series on Computational Intelligence (SSCI), 2016, pp. 1-8, doi: 10.1109/SSCI.2016.7850079.
                        
                        ## A Lexical Approach for Classifying Malicious URLs
                        
                        Darling, Michael. "A Lexical Approach for Classifying Malicious URLs
                        
                        ## PhishDef: URL names say it all
                        
                        A. Le, A. Markopoulou and M. Faloutsos, "PhishDef: URL names say it all," 2011 Proceedings IEEE INFOCOM, 2011, pp. 191-195, doi: 10.1109/INFCOM.2011.5934995.
                        
                        ## Detecting Malicious URLs Using Lexical Analysis
                        
                        Mamun, Mohammad & Rathore, Muhammad & Habibi Lashkari, Arash & Stakhanova, Natalia & Ghorbani, Ali. (2016).
                        Detecting Malicious URLs Using Lexical Analysis. 9955. 467-482. 10.1007/978-3-319-46298-1_30. 
                        
                      '''))
)

make_prediction_layout = html.Div(id='body-container',
                                  children=[
                                    html.H2(children='Choose a model to make a prediction: '),
                                    dcc.Dropdown(id='dropdown_list_3',
                                    options= [{'label':'Random Forest','value':'model_rf'},
                                    {'label':'Gradient Boost','value':'model_gb'},
                                    {'label':'Ridge Classifier','value':'model_rc'}],
                                    value = 'model_rf'),
                                    html.H2(children='Enter a URL to make a prediction:'),
                                    dcc.Input(id='input-on-submit', type='text', value='www.example.com'),
                                    html.Button('Submit', id='submit-val', n_clicks=1),
                                    html.Div(id='prediction_container')
                                    
                                  ])

@app.callback(
    Output('prediction_container', 'children'),
    Input('dropdown_list_3','value'),
    Input('submit-val', 'n_clicks'),
    State('input-on-submit', 'value')
)

def make_prediction(model, n_clicks, url):
    if model == 'model_rf':
        sample_row = functions.url_parse(url)
        text = model_rf.predict(sample_row)[0]
        return 'The Random Forest model predicts that "{}" is {}. This is prediction number {}.'.format(
            url,
            text,
            n_clicks
        )
    elif model == 'model_gb':
        sample_row = functions.url_parse(url)
        text = model_gb.predict(sample_row)[0]
        return 'The Gradient Boost model predicts that "{}" is {}. This is prediction number {}.'.format(
            url,
            text,
            n_clicks
        )
    elif model == 'model_rc':
        sample_row = functions.url_parse(url)
        text = model_rc.predict(sample_row)[0]
        return 'The Ridge Classifier model predicts that "{}" is {}. This is prediction number {}.'.format(
            url,
            text,
            n_clicks
        )



@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/about':
        return about_layout
    elif pathname == '/featureanalysis':
        return feat_analysis_layout
    elif pathname == '/featurelibrary':
        return feat_library_layout
    elif pathname == '/modelanalysis':
        return model_analysis_layout
    elif pathname =='/prediction':
        return make_prediction_layout
    elif pathname == '/references':
        return references_layout
    else:
        return about_layout
    
if __name__ == '__main__':
    app.run_server(debug=True)