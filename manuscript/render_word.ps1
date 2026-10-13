param(
    [string]$DocumentPath = (Join-Path $PSScriptRoot 'GNSS多类型异常识别论文初稿_结构优化版.docx'),
    [string]$OutputDirectory = (Join-Path $PSScriptRoot 'working/render-revision')
)

$ErrorActionPreference = 'Stop'
$paperDocumentPath = [System.IO.Path]::GetFullPath($DocumentPath)
$paperOutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
[void][System.IO.Directory]::CreateDirectory($paperOutputDirectory)
$paperPdfPath = Join-Path $paperOutputDirectory 'paper.pdf'
$paperWord = $null
$paperDocument = $null
try {
    $paperWord = New-Object -ComObject Word.Application
    $paperWord.Visible = $false
    $paperWord.DisplayAlerts = 0
    $paperDocument = $paperWord.Documents.Open($paperDocumentPath, $false, $true)
    $paperDocument.Repaginate()
    $paperDocument.ExportAsFixedFormat($paperPdfPath, 17)
    Write-Output "Exported $($paperDocument.ComputeStatistics(2)) pages to $paperPdfPath"
} finally {
    if ($null -ne $paperDocument) {
        $paperDocument.Close(0)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($paperDocument)
    }
    if ($null -ne $paperWord) {
        if ($paperWord.Documents.Count -eq 0) {
            $paperWord.Quit()
        }
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($paperWord)
    }
}
