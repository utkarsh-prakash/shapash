import seaborn as sns
from shapash.utils.load_smartpredictor import load_smartpredictor
from shapash_modelling import load_data, preprocess

if __name__ == "__main__":
    predictor_load = load_smartpredictor('shapash_model/predictor.pkl')
    df = load_data()
    y=df['tip']
    X=df[df.columns.difference(['tip'])]
    X = preprocess(X)
    predictor_load.add_input(x=X, ypred=y)
    detailed_contributions = predictor_load.detail_contributions()
    predictor_load.modify_mask(max_contrib=3)
    explanation = predictor_load.summarize()
    detailed_contributions.to_excel("shapash_explaination/detailed_contribution.xlsx", index=False)
    explanation.to_excel("shapash_explaination/explanation.xlsx", index=False)
    