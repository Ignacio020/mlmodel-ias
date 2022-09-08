# Initial configuration
import json
config = 'config.json'
with open(config, 'r') as config_json:
# JSON obj as dictionary
    config_data = json.load(config_json)
    columns_of_interest = config_data['columns_of_interest'] 
    columns_to_vectorize = config_data['columns_to_vectorize'] 
    output_features = config_data['output_columns']


#Import libraries and modules
import pandas as pd
from _vectorizer import vectorizer
from pre_process import NLP_ETL
from sklearn.metrics.pairwise import cosine_similarity


class Ticket(pd.DataFrame):
  """
  Custom class for getting recommendations for a given ticket based on similarity.

  Args:
      pd.DataFrame (pandas.DataFrame):  single row Data Frame containing ticket's' information to get recommendation.
  """
    
  def __init__(self,df):
    """
    Object constructor. Returns error message if given Data Frame's lenght is not a single row.
    """
    if len(df) == 1:
      super().__init__(df)
    else:
      print('Error: Wrong dataframe dimension. Dataframe must have a single row')


  def search_similar_problems(self, model, vector_matrix, columns_to_vectorize = columns_to_vectorize,number_of_recomendations = 5):
    """
    Method for searching most similar tickets to the input.

    Args:
        model (KeyedVectors object): mapping of all terms in vocabulary and its corresponding vectors.
        vector_matrix (pandas.DataFrame): two column Data Frame containing ticket number and its corresponding ticket-vector .
        columns_to_vectorize (list [str], optional): list of names of columns to vectorize. See 'config.json' to set 'columns_to_vectorize' default values.
        number_of_recomendations (int, optional): size of recommendation ranking. Defaults to 5.

    Returns:
        Data Frame (pandas.DataFrame): Data Frame of top ranked tickets based on similarity to the input ticket.

    """
  
    ticket = NLP_ETL(self)
    ticket.nlp_superPipeline(columns_to_vectorize)
    ticket.combined_features_tokens()
    self['vector'] = self['tokens_list'].apply(vectorizer,model = model)

    vector_matrix_aux = vector_matrix.copy()
    vector_matrix_aux['sim'] = cosine_similarity(vector_matrix_aux.vector.tolist(), self.vector.to_list())
    return vector_matrix_aux[~vector_matrix_aux['ticket'].isin(self['ticket'])].sort_values('sim',ascending=False).head(number_of_recomendations)

  def attrs_similar_problems(self,similar_tickets,consolidated_data):
    """
    Method for joining USD data to similar_tickets Data Frame. Adds 'similar_problems' as object's attribute.

    Args:
        similar_tickets (pandas.DataFrame): Data Frame of top ranked tickets based on similarity to the input ticket (output of 'Ticket.search_similar_problems' method).
        consolidated_data (pandas.DataFrame): ticket information from USD consolidated database.

    Returns:
        similar_problems (pandas.DataFrame): top similar tickets with USD data.
    """
    similar_problems = similar_tickets.merge(consolidated_data[columns_of_interest],on = 'ticket')
    self.similar_problems = similar_problems
    return similar_problems

  def predict (self, model, vector_matrix, columns_to_vectorize = columns_to_vectorize,number_of_recomendations = 5):
    """
    Method for searching most similar tickets to the input.

    Args:
        model (KeyedVectors object): mapping of all terms in vocabulary and its corresponding vectors.
        vector_matrix (pandas.DataFrame): two column Data Frame containing ticket number and its corresponding ticket-vector .
        columns_to_vectorize (list [str], optional): list of names of columns to vectorize. See 'config.json' to set 'columns_to_vectorize' default values.
        number_of_recomendations (int, optional): size of recommendation ranking. Defaults to 5.

    Returns:
        Similarity map (*.json): map of top similar tickets' numbers and their score.
    """

    ticket = NLP_ETL(self)
    ticket.nlp_superPipeline(columns_to_vectorize)
    ticket.combined_features_tokens()
    self['vector'] = self['tokens_list'].apply(vectorizer, model = model)
    vector_matrix_aux = vector_matrix.copy()
    vector_matrix_aux['sim'] = cosine_similarity(vector_matrix_aux.vector.tolist(), self.vector.to_list())
    sim_rank = vector_matrix_aux[~vector_matrix_aux['ticket'].isin(self['ticket'])].sort_values('sim',ascending=False).head(number_of_recomendations)[['ticket', 'sim']].to_json(indent = 4, orient = 'records')
    #sim_rank = json.loads(sim_rank)
    return sim_rank
  

    # {‘split’, ‘records’, ‘index’, ‘columns’, ‘values’, ‘table’}
  
  def __str__(self):
    """
    Method for printing results in markdown format
    """
    try:
      return self.similar_problems.to_markdown(index=False,tablefmt='pretty')
    except:
      return super().__str__()