import os, json

#file_contains = ['result0','result1','results_gree']
file_contains = ['results_lp','result_lpx']

cases = []
for file in os.listdir():
    if any([file.startswith(fc) for fc in file_contains]):
        with open(file, 'r') as f:
            for content in f.readlines():
                json_content = json.loads(content.replace("'",'"').replace("None",'"None"'))
                print(f"{file} => {json_content['case']}, {json_content['value']}, {json_content['duration']}")
                cases.append(json_content)
ins=1
print("Instance,Campaigns,Customers,Channels,Days")
for jc in sorted(cases, key=lambda c:c['case']['U']):
   case = jc['case']
   value = jc['value']
   print(f"{ins},{value}")
   ins=ins+1

