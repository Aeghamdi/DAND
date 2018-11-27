#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project: Investigate a Dataset (TMDB movie data)
# <a id='home11'></a>
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#limitation">Limitation</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# 
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# In this Project i choose to investigate the TMDB movie data set which contains information about 10,000 movies collected from
# The Movie Database (TMDb),
# including user ratings, revenue, cast members, runtime, release year,... etc.
# 
# **Questions about this data set :**
# 
# <ul>   
# <li>which movie has the most and least profit ?</li>
# <li>which movie has the most and least budgets ?</li>
# <li>which movie has the most and least revenue ?</li>
# <li>what is the average runtime of the movies ?</li>
# <li>who are the top 3 directors with the most released movies ?</li>
# 
# <li>Which genre were more successful?</li>
# 
# <li>List the actors that are most frequent among others</li>
# <li>What kind of properties are associated with movies with high profit?</li>
# </ul>
# 
# <a href="#home11">back to Table of Contents</a>
# 

# In[26]:


# Use this cell to set up import statements for all of the packages that you
#   plan to use.

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html
import pandas as pd
import numpy as np
import seaborn as sns
import csv
from datetime import datetime
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# ### General Properties

# In[2]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
tmdb = pd.read_csv('tmdb-movies.csv')
tmdb.head()


# **I noticed the following :**
# 
# <ul>   
# <li>The unit of currency is not mentioned in this data set so we will assume it is US dollar.</li>
# 
# <li>The popularity is not correct because the voters count diverse for each movie so it's not valid to be used in analysing.</li>
# </ul>
# 
# ### Data Cleaning (Replace this with more specific notes!)
# 
# **i will do the following :**
# 
# <ul>   
# <li>Removing columns such as: id, imdb_id, vote_count, production_company, keywords, homepage.</li>
# <li>Removing the duplicacy.</li>
# <li>Discarding records that have zero budget or zero revenue.</li>
# <li>Converting release date column into date format.</li>
# <li>Replacing zero with NAN in runtime column.</li>
# <li>Changing format of budget and revenue column.</li>
# </ul>
# 
# <a href="#home11">back to Table of Contents</a>
# 

# In[3]:


#Removing columns such as: id, imdb_id, vote_count, production_company, keywords, homepage
del_col=[ 'id', 'imdb_id', 'popularity', 'budget_adj', 'revenue_adj', 'homepage', 'keywords', 'overview', 'vote_count', 'vote_average']
tmdb= tmdb.drop(del_col,1)
tmdb.head()


# In[6]:


# counting rowa and columns 
rows, col = tmdb.shape
print('There number of movies is {} movie and the no.of columns is {} column '.format(rows-1, col))


# In[7]:


#Removing the duplicacy.
tmdb.drop_duplicates(keep ='first', inplace=True)
rows, col = tmdb.shape

# counting rowa and columns again 
print('There number of movies is {} movie and the no.of columns is {} column '.format(rows-1, col))


# In[8]:


#Discarding records that have zero budget or zero revenue.
list11=['budget', 'revenue']
tmdb[list11]=tmdb[list11].replace(0,np.NAN)
tmdb.dropna(subset = list11, inplace = True)

rows, col = tmdb.shape
print('I discarded the The records with zero budget or zero revenue, \nso now the number of movies is {} movie and the no.of columns is {} column '.format(rows-1, col))


# In[9]:


#Converting release date column into date format.
tmdb.release_date = pd.to_datetime(tmdb['release_date'])

#to check the affected recoreds
tmdb.head()


# In[10]:


#Replacing zero with NAN in runtime column.
tmdb['runtime'] =tmdb['runtime'].replace(0, np.NAN)

#checking the current datatypes
tmdb.dtypes


# In[11]:


#Changing format of budget and revenue column.
change_type=['budget','revenue']
tmdb[change_type]=tmdb[change_type].applymap(np.int64)

#checking the current datatypes
tmdb.dtypes


# <a href="#home11">back to Table of Contents</a>

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# 
# ### Research Question 1 
# **which movies has the most and least profit ?**
# 
# 
# in order to know this i will calculate the profit for each movie then assign it to new column in the dataset.

# In[12]:


#calculate the profit and assign it to new column
tmdb.insert(2,'profit',tmdb['revenue']-tmdb['budget'])

tmdb.head(2)


# in order to know the min & max of any column i will create new function.

# In[13]:


def simple_analysis(column):

    highest= tmdb[column].idxmax()
    high11=pd.DataFrame(tmdb.loc[highest])

    lowest= tmdb[column].idxmin()
    low11=pd.DataFrame(tmdb.loc[lowest])

    result=pd.concat([high11, low11], axis=1)
    
    return result

simple_analysis('profit')


# ### Answear Question 1 
# which movie has the most and least profit ?
# 
# **The result shows that "Avatar" movie has the highest profit and "The Warrior's Way" has the lowest.**
# 
# <a href="#home11">back to Table of Contents</a>

# ### Research Question 2 
# **which movie has the most and least budgets ?**
# 
# 
# in order to know this i will use the same function, simple_analysis.

# In[14]:


simple_analysis('budget')


# ### Answear Question 2 
# which movie has the most and least budgets ?
# 
# **The result shows that "The Warrior's Way" movie has the highest budget and "Lost & Found" has the lowest.**
# 
# <a href="#home11">back to Table of Contents</a>

# ### Research Question 3
# **which movie has the most and least revenue ?**
# 
# 
# in order to know this i will use the same function, simple_analysis.

# In[15]:


simple_analysis('revenue')


# ### Answear Question 3
# which movie has the most and least revenue ?
# 
# **The result shows that "Avatar" movie has the highest revenue and "Shattered Glass" has the lowest.**
# 
# <a href="#home11">back to Table of Contents</a>

# ### Research Question 4 
# 
# **what is the average runtime of the movies ?**
# 
# in order to know this i will create function to calculate the average of all the values in the given column.

# In[16]:


def average(column):
    return tmdb[column].mean()


# now i will run the function with the "Runtime coulmn"
# 
# ### Answear Question 4
# what is the average runtime of the movies ?

# In[17]:


avg11=average('runtime')
print('The average runtime of all the movies is {} minutes '.format(round(avg11, 5)))


# **Representing the average data in a histogram**

# In[18]:


plt.figure(figsize=(9,5), dpi = 100)

plt.xlabel('Runtime in minutes', fontsize = 13)
plt.ylabel('Movies', fontsize=13)
plt.title('Runtime of all the movies', fontsize=13)
plt.hist(tmdb['runtime'], rwidth = 0.8, bins =37, color='red')

plt.show()


# <a href="#home11">back to Table of Contents</a>

# ### Research Question 5 
# 
# **who are the top 3 directors with the most released movies ?**
# 
# in order to know this i must create a function to separate the content because some columns contains more that one value separated by '|' 

# In[19]:


def separated_by(column_name):
    a = tmdb[column_name].str.cat(sep = '|')
    a = pd.Series(a.split('|'))
    #desc
    count = a.value_counts(ascending = False)
    
    return count


# now i will use this (separated_by) function.

# ### Answear Question 5
# who are the top 3 directors with the most released movies ?

# In[23]:


directors = separated_by('director')

directors.head(3)


# <a href="#home11">back to Table of Contents</a>

# ### Research Question 6
# 
# **Which genre were more successful?**
# 
# in order to know this i will call my function separated_by

# In[36]:


genres = separated_by('genres')


# ### Answear Question 6
# Which genre were more successful?

# In[86]:


genres.head(4)


# **Representing the genres data in a graph**

# In[87]:


genres.plot(kind= 'barh',figsize = (13,6),fontsize=12,colormap='tab20c')

plt.title("Most Popular Genres",fontsize=15)
plt.xlabel('Number Of Movies',fontsize=15)
plt.ylabel("Genres",fontsize= 15)
sns.set_style("whitegrid")


# <a href="#home11">back to Table of Contents</a>

# ### Research Question 7
# 
# **List the actors that are most frequent among others**
# 
# in order to know this i will call my function separated_by

# In[39]:


actors = separated_by('cast')


# ### Answear Question 7
# List the actors that are most frequent among others

# In[42]:


actors.head(12)


# **Representing the actors data in a graph**

# In[60]:


actors.iloc[:12].plot.bar(figsize=(13,6),colormap= 'tab20c',fontsize=12)

#setup the title and the labels of the plot.
plt.title("Most Frequent Actor",fontsize=15)
plt.xticks(rotation = 70)
plt.xlabel('Actor',fontsize=13)
plt.ylabel("Number Of Movies",fontsize= 13)
sns.set_style("whitegrid")


# <a href="#home11">back to Table of Contents</a>

# ### Research Question 8
# 
# **What kind of properties are associated with movies with high profit?**
# 
# In order to analyze this i will compare and try to figure the relation between the Profit and (runtime ,Budget)

# In[85]:



relation1 = pd.DataFrame(tmdb['profit'].sort_values(ascending=False))
data_set1 = ['budget','runtime']
for i in data_set1:
    relation1[i] = tmdb[i]
relation1.head(10)


# In[83]:


fig, axes = plt.subplots(2,figsize = (16,6))
fig.suptitle("Profit Vs (Budget,Runtime)",fontsize=14)


sns.regplot(x=tmdb['profit'], y=tmdb['budget'],color='c',ax=axes[0])
sns.regplot(x=tmdb['profit'], y=tmdb['runtime'],color='c',ax=axes[1])


sns.set_style("whitegrid")


# ### Answear Question 8
# What kind of properties are associated with movies with high profit?
# 
# **Profit & Budget :**
# It turn out that the more you spend in the movie the chance of the profit to be increased is getting more too.
# From the graph and the line i can tell that the percentage of this to happen can be between 50% and 70%.
# 
# **Profit & Runtime :**
# I can not tell that there is a real relation here, i can not guarantee at all, so runtime of the movie will not affect the profit.
# 
# it might happen but it suly will be caused by other factors as the main causes of it.
# 
# **I can tell that the affect of the Budget is greater than the effect of the Runtime.**

# <a href="#home11">back to Table of Contents</a>

# <a id='limitation'></a>
# ## Limitations or Challenges
# 
# **1-The data contained some extra column that are not sutable for the analysis purpose such as : id, imdb_id, keywords, homepage.**
# 
# **2-Some columns has a null ( NAN ) values.** 
# 
# in this case you either delete it or replace it with a value, for example : the mean of the entire column can be great choice.
# 
# **2-Some columns has wrong data type :so i had to convert release date column into date format** 
# 
# **3-Some records had zero budget or zero revenue, this is clearly unuseful data so i removed it, it was around 7011 records, its almost 70% of the recored so the dataset became small after this removing, thats why i can say the dataset had a real lack of accuracy when it comes to the budget and revenue columns.**
# 
# **4-The columns of the dataset perresents money but without mentioning the currency, i assumed it and used US dollar but if we want to do the analysis process compared to another makert rather that US the currency is a must.**
# 
# 

# <a href="#home11">back to Table of Contents</a>

# <a id='conclusions'></a>
# ## Conclusions
# 
# **The data was huge and very useful, it can be a great material to analyse more deep in order to get the effect of a specific factor, for example :**
# 
# What factors can lead the movie to be more popular or to have a high vote rating.
# 
# **I consider my work as the minimum effort on such dataset.**
# 
# **Conclusions:**
# 
# **1- The average runtime of all the movies is 109.22029 minutes**
# 
# **2- top 3 directors with the most released movies are :**
# 
#     Steven Spielberg    28 movies
#     Clint Eastwood      24 movies
#     Ridley Scott        21 movies
#     
# **3- Drama is the mots popular Genres, it is the first with 1756 movies,it is surly the producer's faviorate Genre among other Genres, followed by :** 
# 
#     Comedy with      1358 movies
#     Thriller with    1204 movies
#     Action with      1085 movies
#     
# **4- Robert De Niro frequently appears more than oter actors with 52 movies, followed by :**
# 
#     Bruce Willis          46 movies
#     Samuel L. Jackson     44 movies
#     Nicolas Cage          43 movies
# 
# **5- There is a relation between the Profit & the Budget althought its not strong one but the more you spend in the movie the chance of the profit to be increased is getting more too.**
# 
# **6- There was not a clear affect of the Runtime on the Profit of the movie.**
# 
# 
#  
# 
#  
# 
#  
#  
# **Refrences :**
# 
# https://matplotlib.org/users/pyplot_tutorial.html
# 
# https://matplotlib.org/tutorials/colors/colormaps.html
# 
# http://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html
# 
# https://docs.scipy.org/doc/numpy/user/quickstart.html
# 
# https://carlyhochreiter.files.wordpress.com/2018/05/investigating-movie-dataset.pdf
# 
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html
# 
# https://seaborn.pydata.org/tutorial/aesthetics.html?highlight=set_style
# 
# https://seaborn.pydata.org/generated/seaborn.regplot.html?highlight=regplot
# 
# http://pandas.pydata.org/pandas-docs/stable/indexing.html
# 
# https://matplotlib.org/users/pyplot_tutorial.html
# 
# Python Data Science Handbook by Jake VanderPlas : Chapter 4. Visualization with Matplotlib

# <a href="#home11">back to Table of Contents</a>

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

