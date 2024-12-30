import streamlit as st
import requests
from bs4 import BeautifulSoup

def get_schedule():
    print("Gathering Schedule Data...")
    # Make a request to the website
    r = requests.get('https://www.fftoday.com/nfl/schedule.php')
    r_html = r.text

    # Create a BeautifulSoup object and specify the parser
    soup = BeautifulSoup(r_html, 'html.parser')

    # Find the table in the HTML
    table = soup.find('table', attrs={'width': '80%', 'border': '0', 'cellpadding': '0', 'cellspacing': '0'})

    # Find all rows in the table with a white background
    rows = table.find_all('tr', attrs={'bgcolor': '#ffffff'})
    print("Schedule Data Retrieved")
    return table, rows


st.title("NFL Schedule Viewer")

if st.button("Get Schedule"):
    table, rows = get_schedule() # Call the function
    
    if table:
        st.write("Schedule Table (HTML):")
        st.markdown(str(table), unsafe_allow_html=True) #Display table as HTML
    else:
         st.write("Error. Could not find the table.")
         
    if rows:
        st.write(f"Number of Schedule Rows: {len(rows)}") #Display row length
    else:
         st.write("Error. Could not find the rows")
