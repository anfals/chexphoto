def process_string(str):
    res = [r.split(":")[1].strip() for r in results.split(";")]
    print("\n".join(res))



results = "Pleural Effusion MCC: 0.721 ; Pleural Effusion AUC: 0.939 ; Pleural Effusion Precision: 0.851 ; Pleural Effusion Recall: 0.836 ; Pleural Effusion f1: 0.843 ; Pleural Effusion Accuracy: 0.863 ; Edema MCC: 0.694 ; Edema AUC: 0.938 ; Edema Precision: 0.828 ; Edema Recall: 0.745 ; Edema f1: 0.784 ; Edema Accuracy: 0.870 ; Atelectasis MCC: 0.513 ; Atelectasis AUC: 0.856 ; Atelectasis Precision: 0.681 ; Atelectasis Recall: 0.624 ; Atelectasis f1: 0.651 ; Atelectasis Accuracy: 0.801 ; Consolidation MCC: 0.517 ; Consolidation AUC: 0.867 ; Consolidation Precision: 0.683 ; Consolidation Recall: 0.512 ; Consolidation f1: 0.585 ; Consolidation Accuracy: 0.870 ; Cardiomegaly MCC: 0.677 ; Cardiomegaly AUC: 0.944 ; Cardiomegaly Precision: 0.767 ; Cardiomegaly Recall: 0.690 ; Cardiomegaly f1: 0.727 ; Cardiomegaly Accuracy: 0.914 ; MCC Average: 0.624 ; AUC Average: 0.909 ; Overall Accuracy: 0.863 ; loss: 0.316"
process_string(results)