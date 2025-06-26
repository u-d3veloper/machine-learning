def main():
    import pandas as pd
    import matplotlib.pyplot as plt
    fruits = pd.read_table('fruit_data_with_colors.txt')
    print(fruits.head())
    print(fruits['fruit_name'].unique())
    print(fruits.groupby('fruit_name').size())
    
if __name__ == "__main__":
    main()
