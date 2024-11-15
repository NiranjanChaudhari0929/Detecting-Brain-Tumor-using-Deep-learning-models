# Plot the confusion matrix as a heatmap
class_names = ['Meningioma', 'Glioma', 'Pituitary Tumor']
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
plt.figure(figsize=(6, 4))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print the classification report
print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# Evaluate the model on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Calculate micro-average precision, recall, and F1 score
precision_micro = precision_score(y_test_classes, y_pred_classes, average='micro')
recall_micro = recall_score(y_test_classes, y_pred_classes, average='micro')
f1_micro = f1_score(y_test_classes, y_pred_classes, average='micro')

# Compute micro-average test accuracy
test_accuracy_micro = accuracy_score(y_test_classes, y_pred_classes)

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(x_val, y_val)

print(f'\nMicro-average Precision: {precision_micro:.4f}')
print(f'Micro-average Recall: {recall_micro:.4f}')
print(f'Micro-average F1 Score: {f1_micro:.4f}')
print(f'Micro-average Test Accuracy: {test_accuracy_micro:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')
print(f'Micro-average Test Loss: {val_loss:.4f}')
