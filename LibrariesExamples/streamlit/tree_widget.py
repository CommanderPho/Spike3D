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
# def build_tree(tree, node):
#     # If the node is a dictionary, iterate through items
#     if isinstance(node, dict):
#         for key, value in node.items():
#             # Use a checkbox to represent the collapsible parent node
#             if tree.checkbox(f"{key}", key):
#                 # Recursively build the tree for the child node (likely a list of files)
#                 build_tree(tree, value)
#     # If the node is a list, display the items
#     elif isinstance(node, list):
#         for item in node:
#             # You can replace this with a file display or download link
#             tree.write(f"- {item}")


def build_tree(tree, path, node):
    # If the node is a dictionary, iterate through items
    if isinstance(node, dict):
        for key, value in node.items():
            new_path = f"{path}/{key}"
            tree.write(f"- {new_path}")
            # Use a checkbox to represent the collapsible parent node
            # if tree.checkbox(f"{key}", new_path):
            #     # Recursively build the tree for the child node (likely a list of files)
            #     build_tree(tree, new_path, value)

            # Recursively build the tree for the child node (likely a list of files)
            build_tree(tree, new_path, value)


    # If the node is a list, display the items
    elif isinstance(node, list):
        for item in node:
            new_path = f"{path}/{item}"

            # You can replace this with a file display or download link
            # tree.write(f"- {item}")
            tree.checkbox(f"{new_path}", item)



# Call the function to build the tree starting with the sessions_dict
st.sidebar.title('Session File Tree')
build_tree(tree, '/', sessions_dict)
