import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

# read transaction data
store_data = pd.read_csv("store_data.csv", header=None)

transactions = []
for i in range(0, store_data.shape[0]):
    transaction = []
    for j in range(0, store_data.shape[1]):
        if str(store_data.values[i, j]) != "nan":
            transaction.append(str(store_data.values[i, j]))
    transactions.append(transaction)

# convert transactions to dataframe
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# fpgrowth
# frequent itemsets with min_support=0.0045
frequent_itemsets = fpgrowth(df, min_support=0.0045, use_colnames=True)

# association rules with confidence min_threshold=0.2
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

# list top 10 rules with confidence
top_10_confidence_rules = rules.sort_values(by="confidence", ascending=False).head(10)
print("Top 10 Confidence Rules:")
print(top_10_confidence_rules)

# list top 10 rules with lift
top_10_lift_rules = rules.sort_values(by="lift", ascending=False).head(10)
print("\nTop 10 Lift Rules:")
print(top_10_lift_rules)
