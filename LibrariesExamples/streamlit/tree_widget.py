import streamlit as st

# Example dictionary (session names and their files)
sessions_dict = {
    'Session A': ['file1.txt', 'file2.txt', 'file3.txt'],
    'Session B': ['file1.txt', 'file4.txt'],
    'Session C': ['file5.txt', 'file6.txt'],
}

# Initialize a container to hold the tree
tree = st.sidebar.container()

# Recursive function to build a tree given a node (dict or list)
def build_tree(tree, node):
    # If the node is a dictionary, iterate through items
    if isinstance(node, dict):
        for key, value in node.items():
            # Use a checkbox to represent the collapsible parent node
            if tree.checkbox(f"{key}", key):
                # Recursively build the tree for the child node (likely a list of files)
                build_tree(tree, value)
    # If the node is a list, display the items
    elif isinstance(node, list):
        for item in node:
            # You can replace this with a file display or download link
            tree.write(f"- {item}")

# Call the function to build the tree starting with the sessions_dict
st.sidebar.title('Session File Tree')
build_tree(tree, sessions_dict)