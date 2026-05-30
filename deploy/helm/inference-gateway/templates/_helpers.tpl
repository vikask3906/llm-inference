{{/* Standard fullname / labels helpers, K8s-idiomatic. */}}

{{- define "inference-gateway.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "inference-gateway.fullname" -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "inference-gateway.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "inference-gateway.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: gateway
{{- end -}}

{{- define "inference-gateway.selectorLabels" -}}
app.kubernetes.io/name: {{ include "inference-gateway.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "inference-gateway.backendsString" -}}
{{- $pairs := list -}}
{{- range .Values.backends -}}
{{- $pairs = append $pairs (printf "%s=%s" .id .url) -}}
{{- end -}}
{{- join "," $pairs -}}
{{- end -}}
