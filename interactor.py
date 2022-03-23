import pandas as pd
import pyperclip as pc

df = pd.read_excel("IR1_7k_news.xlsx")

while True:
    pc.copy(df["content"][int(input("Enter id: "))])
