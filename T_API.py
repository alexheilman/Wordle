import tweepy
import pandas as pd
import matplotlib.pyplot as plt

consumer_key = #
consumer_secret = #
bearer_token = #
access_token = #
access_secret = #

day = 274

correct_letter = '🟩'
wordle_day = 'Wordle ' + str(day)

client = tweepy.Client(bearer_token)
#client.search_recent_tweets('covid OR covid19 is:retweet -is:retweet has:media -has:media from:x')
query = wordle_day + ' -is:retweet'
response = client.search_recent_tweets(query=query, max_results=10)
response2 = tweepy.Paginator(client.search_recent_tweets, query=query, max_results=10).flatten(limit=1000)

df = pd.DataFrame(columns = ['completion', '1_pct', '2_pct', '3_pct', '4_pct', '5_pct', '6_pct'])

for tweet in response2:#.data:
    df = df.reindex(df.index.tolist() + list(range(df.shape[0], df.shape[0]+1)))

    #temp = []
    #temp.append(tweet.text.find('⬛'))
    #temp.append(tweet.text.find('⬜'))
    #temp.append(tweet.text.find('🟨'))
    #temp.append(tweet.text.find('🟩'))

    #l = min([i for i in temp if i >= 0])
    #r = max(tweet.text.rfind('⬛'), tweet.text.rfind('⬜'), tweet.text.rfind('🟨'), tweet.text.rfind('🟩'))
    
    #print('----------------------------')
    #print((r-l+2)/6)
    #print(r)
    #print(tweet.text)
    x = tweet.text.find('/6')
    df.iloc[-1, 0] = tweet.text[x-1]

#temp = df['completion'].value_counts()
#temp.plot(kind='bar').savefig('here.png')

counts = df['completion'].value_counts().reindex(['1','2','3','4','5','6','X'], fill_value = 0 )

plt.bar(counts.index.values, height=counts.values, color='blue', edgecolor='black')
plt.title('Wordle Completion Attempts \n' + wordle_day)
plt.xlabel('# of Attempts')





plt.savefig('Second Attempt Results.png', bbox_inches='tight')

#print(df)
#print(response)

#auth = tweepy.OAuthHandler(API_key, API_secret_key)
#auth.set_access_token(auth)