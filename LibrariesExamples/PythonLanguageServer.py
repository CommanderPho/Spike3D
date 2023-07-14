from lspclient import lspclient

# Connect to the language server
client = lspclient.LanguageClient('python')

# Get the call hierarchy for a function
call_hierarchy = client.textDocument.callHierarchy(
    uri='file:///path/to/your/file.py',
    position={'line': 10, 'character': 5}
)

# Print the callers and callees of the function
for item in call_hierarchy:
    print(item['name'])
    print('Callers:')
    for caller in item['incomingCalls']:
        print(caller['from']['name'])
    print('Callees:')
    for callee in item['outgoingCalls']:
        print(callee['to']['name'])