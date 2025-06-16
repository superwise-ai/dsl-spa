import pandas as pd

def add_date_information(df: pd.DataFrame, year_field: str, quarter_field: str, month_field: str, sort_by_date: bool = True) -> pd.DataFrame:
    """Adds a quarter column to df. Sorts the data if sort_by_data is true.

    Args:
        df (pd.DataFrame): Input dataframe
        year_field (str): Name of year column
        quarter_field (str): Desired name of quarter column
        month_field (str): Name of month column
        sort_by_date (bool, optional): Whether to sort the data by year/month ascending. Defaults to True.

    Returns:
        pd.DataFrame: Output dataframe
    """
    if len(df) <= 0:
        return df
    df[quarter_field] = df[month_field].apply(lambda x: (x-1)//3 + 1)
    if sort_by_date:
        df = df.sort_values(by=[year_field,month_field])
    return df

def build_time_series_counts(df,group_by: str, year_field: str, quarter_field: str, month_field: str, count_field: str, result_field: str, start_date: str = None, end_date: str = None, date_field: str = "x") -> pd.DataFrame:
    """Creates counts of records based on group_by value.

    Args:
        df (pd.DataFrame): Input dataframe
        group_by (str): The aggregation to group data by. Must be year, quarter, or month.
        year_field (str): Name of year column
        quarter_field (str): Name of quarter column
        month_field (str): Name of month column
        count_field (str): Name of count column. If None it treats every row as a count of 1.
        result_field (str): Name of column to store sum of counts to.
        start_date (str, optional): Date to start aggregation on. If None infers from data. Defaults to None.
        end_date (str, optional): Date to end aggregation on. If None infers from data. Defaults to None.

    Returns:
        pd.DataFrame: Output
    """
    if len(df) <= 0:
        return df
    if start_date is None:
        first_year = df[year_field].values[0]
        first_quarter = df[quarter_field].values[0]
        first_month = df[month_field].values[0]
    else:
        first_year = int(start_date[:4])
        first_month = int(start_date[5:7])
        first_quarter = (first_month-1)//3 + 1
    if end_date is None:
        last_year = max(df[year_field].values)
        last_quarter = None
        last_month = None
    else:
        last_year = int(end_date[:4])
        last_month = int(end_date[5:7])
        last_quarter = (last_month-1)//3 + 1

    graph_data_x = []
    graph_data_y = []
    if group_by == "year":
        for year in range(first_year,last_year+1):
            partial_data = df[df[year_field] == year]
            if len(partial_data.index) <= 0:
                sum_count = 0
            else:
                sum_count = sum(partial_data[count_field])
            graph_data_x.append(str(year))
            graph_data_y.append(sum_count)
    elif group_by == "quarter":
        for year in range(first_year,last_year+1):
            begin_quarter = 1
            end_quarter = 4
            partial_year_data = df[df[year_field] == year]
            if year == first_year:
                begin_quarter = first_quarter
            elif year == last_year:
                if last_quarter is None:
                    end_quarter = max(partial_year_data[quarter_field].values)
                else:
                    end_quarter = last_quarter
            for quarter in range(begin_quarter,end_quarter+1):
                partial_data = partial_year_data[partial_year_data[quarter_field] == quarter]
                if len(partial_data.index) <= 0:
                    sum_count = 0
                else:
                    sum_count = sum(partial_data[count_field])
                graph_data_x.append(f"{year}-Q{quarter}")
                graph_data_y.append(sum_count)
    elif group_by == "month":
        for year in range(first_year,last_year+1):
            begin_month = 1
            end_month = 12
            partial_year_data = df[df[year_field] == year]
            if year == first_year:
                begin_month = first_month
            elif year == last_year:
                if last_month is None:
                    end_month = max(partial_year_data[month_field].values)
                else:
                    end_month = last_month
            for month in range(begin_month,end_month+1):
                partial_data = partial_year_data[partial_year_data[month_field] == month]
                if len(partial_data.index) <= 0:
                    sum_count = 0
                else:
                    sum_count = sum(partial_data[count_field])
                graph_data_x.append(f"{year}-{str(month).zfill(2)}")
                graph_data_y.append(sum_count)
    data = {date_field: graph_data_x, result_field: graph_data_y}
    return pd.DataFrame.from_dict(data, orient = 'columns')

def build_pie_graph(df: pd.DataFrame, index_field: str, value_field: str, label_field: str, count_field: str = None) -> pd.DataFrame:
    """Formats data for a pie graph.

    Args:
        df (pd.DataFrame): Input dataframe
        index_field (str): Categorical column indicating the name of each pie slice
        value_field (str): Name of column to store the size of each pie slice
        label_field (str): Name of column to store the name of each pie slice
        count_field (str, optional): Name of column to sum for calculating the pie slice size. If None treats each row as 1. Defaults to None.

    Returns:
        pd.DataFrame: A dataframe with two columns <label_field> and <value_field>
    """
    if len(df) <= 0:
        return df
    indexes = df[index_field].unique()
    data = []
    for code in indexes:
        partial_data = df[df[index_field] == code]
        if count_field is None:
            data.append(len(partial_data.index))
        else:
            data.append(sum(partial_data[count_field].values))
    return pd.DataFrame.from_dict({value_field: data, label_field: indexes}, orient= "columns")

def build_stacked_bar_graph_from_data_columns(df: pd.DataFrame, index_field, data_fields: list[str], value_field: str, color_field: str) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): Input dataframe
        index_field (_type_): Name of column to use as x data for bar graph
        data_fields (list[str]): List of fields to use as sizes for each stacked bar
        value_field (str): Name of column to store the stacked bar size in
        color_field (str): Name of column to store the label for each stacked bar

    Returns:
        pd.DataFrame: A dataframe with three columns, <index_field>, <value_field>, and <color_field>
    """
    if len(df) <= 0:
        return df
    data = []
    for data_field in data_fields:
        for i,row in df.iterrows():
            index = row[index_field]
            value = row[data_field]
            color = data_field
            data.append([index,value,color])
    df = pd.DataFrame(data,columns=[index_field,value_field,color_field])
    return df

def filtered_arithmetic_column_operation(df: pd.DataFrame, column: str, operator: str, value: float, where_column: float, where_value: float) -> pd.DataFrame:
    """Applies an arithmetic operation to column where the column where_column has value where_value.

    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Name of column to apply operation to
        operator (str): Arithmetic operator to apply (+,-,*,/)
        value (float): Value to use for arithmetic operation
        where_column (float): Column to check for where_value
        where_value (float): Value to check where_column for

    Returns:
        pd.DataFrame: Output dataframe
    """
    if len(df) <= 0:
        return df
    df_where_equals = df[df[where_column].astype(str) == where_value]
    df_where_not_equals = df[df[where_column].astype(str) != where_value]
    if operator == "+":
        df_where_equals.loc[:,column] = df_where_equals.loc[:,column] + value
    elif operator == "-":
        df_where_equals.loc[:,column] = df_where_equals.loc[:,column] - value
    elif operator == "*":
        df_where_equals.loc[:,column] = df_where_equals.loc[:,column] * value
    elif operator == "/":
        df_where_equals.loc[:,column] = df_where_equals.loc[:,column] / value
    df = pd.concat([df_where_equals,df_where_not_equals])
    return df

def arithmetic_combine_column(df: pd.DataFrame, col1: str, col2: str, new_column: str, operation: str) -> pd.DataFrame:
    """Combines two columns together using an arithmetic operation. new_column = col1 (+,-,*,/) col2

    Args:
        df (pd.DataFrame): Input dataframe
        col1 (str): Name of left column
        col2 (str): Name of right column
        new_column (str): Name of new column to store result in
        operation (str): Operation to apply. Must be +,-,*,/.

    Returns:
        pd.DataFrame: Output dataframe
    """
    if len(df) <= 0:
        return df
    new_values = []
    col1_values = df[col1].values
    col2_values = df[col2].values
    if operation in ["+",'-','*']:
        for i in range(len(col1_values)):
            if operation == '+':
                new_values.append(col1_values[i]+(col2_values[i]))
            elif operation == '-':
                new_values.append(col1_values[i]-(col2_values[i]))
            elif operation == '*':
                new_values.append(col1_values[i]*(col2_values[i]))
    else:
        for i in range(len(col1_values)):
            if col2_values[i] == 0:
                new_values.append(0)
            else:
                new_values.append(col1_values[i]/(col2_values[i]))
    df[new_column] = new_values
    return df

def replace_value_in_column(df: pd.DataFrame, column: str, old_value, new_value) -> pd.DataFrame:
    """Replaces all instances of old_value in column with new_value

    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Name of column to search for old_value
        old_value (_type_): value to search column for
        new_value (_type_): value to replace old_value with in column

    Returns:
        pd.DataFrame: Output dataframe
    """
    df[column] = df[column].replace(old_value,new_value)
    return df

def filter_by_value(df: pd.DataFrame, column: str, value) -> pd.DataFrame:
    """Filters df rows where the column is value.

    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Name of column to search for value
        value (_type_): Value to filter by

    Returns:
        pd.DataFrame: Output dataframe
    """
    return df[df[column] == value]

def value_counts(df: pd.DataFrame, column: str, column_name: str = "count") -> pd.DataFrame:
    """Generates the pandas value_counts for a given column

    Args:
        df (pd.DataFrame): Input dataframe
        column (str): Name of column to do value_counts on
        column_name (str, optional): Name of column to save counts to. Defaults to "count".

    Returns:
        pd.DataFrame: New datafarme with one column, column_name, containing the counts of every value.
    """
    series = df[column].value_counts()
    df = series.to_frame(name=column_name)
    return df

def sort_values(df: pd.DataFrame, by: list, ascending: bool = True) -> pd.DataFrame:
    """Sorts df by column by in ascending or descending order.

    Args:
        df (pd.DataFrame): Input dataframe
        by (list): Column to sort on
        ascending (bool, optional): Whether to do ascending or descending sort. Defaults to True.

    Returns:
        pd.DataFrame: Sorted Dataframe
    """
    df = df.sort_values(by=by,ascending=ascending)
    return df

pipeline_functions_dict = {
    "build_time_series_counts": build_time_series_counts,
    "add_date_information": add_date_information,
    "build_pie_graph": build_pie_graph,
    "build_stacked_bar_graph_from_data_columns": build_stacked_bar_graph_from_data_columns,
    "arithmetic_column_operation": filtered_arithmetic_column_operation,
    "arithmetic_combine_column": arithmetic_combine_column,
    "replace": replace_value_in_column,
    "filter_by_value": filter_by_value,
    "value_counts": value_counts,
    "sort_values": sort_values
}