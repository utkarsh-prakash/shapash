import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from shapash.explainer.smart_explainer import SmartExplainer

def load_data():
    df=sns.load_dataset('tips')
    return df

def preprocess(X):
    X['day']=X['day'].cat.codes
    X['sex']=X['sex'].cat.codes
    X['smoker']=X['smoker'].cat.codes
    X['time']=X['time'].cat.codes
    return X

def train_model(X_train, y_train):
    regressor = RandomForestRegressor(n_estimators=200).fit(X_train,y_train)
    return regressor

def compile_shapash_model(x, model):
    xpl = SmartExplainer()
    xpl.compile(
        x=x,
        model=model,
    )
    return xpl

def save_shapash_predictor(xpl):
    predictor = xpl.to_smartpredictor()
    predictor.save('shapash_model/predictor.pkl')

if __name__ == "__main__":
    df = load_data()
    y=df['tip']
    X=df[df.columns.difference(['tip'])]
    X = preprocess(X)
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=1)
    regressor = train_model(X_train, y_train)
    xpl = compile_shapash_model(X_test, regressor)
    save_shapash_predictor(xpl)
    # Run a dash on local server
    app = xpl.run_app(title_story='Tips Dataset', host='127.0.0.1')
