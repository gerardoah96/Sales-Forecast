<?php
/**
 * run_weekly_forecast.php
 *
 * PHP entrypoint that:
 *  - Computes the "as-of" date (today or ?date=YYYY-MM-DD)
 *  - Calls a Python runner that:
 *      1) fetches raw sales from DB (source_query)
 *      2) runs online_simulator (current week + horizon_weeks)
 *      3) deletes existing forecast rows for that predicted window (if any)
 *      4) inserts new forecasts
 *
 * Assumptions:
 *  - Your Python runner accepts argparse flags:
 *      --current_date
 *      --horizon_weeks
 *      --db_host --db_name --db_user --db_pass
 *      --forecast_table
 *      --algo_id
 *      --source_query
 */

// -----------------------------------------------------------------------------
// CONFIG
// -----------------------------------------------------------------------------
$projectRoot = dirname(__DIR__);              // /project-root
$pythonPath  = $projectRoot . '/.venv/bin/python';

// Point this at the python script that does: run + delete + insert.
// If you truly have src/Sim/online_runner.py, keep it.
// If your actual runner is uploader.py, change it accordingly.
$runnerPath  = $projectRoot . '/src/Sim/online_runner.py';
// مثال بديل:
// $runnerPath  = $projectRoot . '/src/uploader.py';

// DB (source + target). If you really have separate source/target DBs,
// your runner must accept separate args; otherwise keep them the same here.
$dbHost   = '127.0.0.1';
$dbName   = 'SALES_DB';
$dbUser   = 'sales_user';
$dbPass   = 'sales_pass';

// Forecast table + algo id
$forecastTable = 'forecast_weekly';
$algoId        = 42;

// Horizon (weeks ahead). 4 => current week + 4 = 5 week-starts total
$horizonWeeks = 4;

// SQL query to fetch raw daily sales that the model expects
$sourceQuery = <<<SQL
SELECT Date, Cadena, Tienda, Producto, Cantidad
FROM sales_table
SQL;

// As-of date (can override via ?date=YYYY-MM-DD)
$currentDate = (isset($_GET['date']) && $_GET['date'] !== '')
    ? $_GET['date']
    : date('Y-m-d');


// -----------------------------------------------------------------------------
// HELPER: run the Python runner
// -----------------------------------------------------------------------------
function run_weekly_simulation(array $opts): array
{
    $projectRoot = rtrim($opts['project_root'], '/');

    $parts = [];
    $parts[] = escapeshellarg($opts['python_path']);
    $parts[] = escapeshellarg($opts['runner_path']);

    // Required args (must match argparse in your Python runner)
    $parts[] = '--current_date '    . escapeshellarg($opts['current_date']);
    $parts[] = '--horizon_weeks '   . escapeshellarg((string)$opts['horizon_weeks']);

    $parts[] = '--db_host '         . escapeshellarg($opts['db_host']);
    $parts[] = '--db_name '         . escapeshellarg($opts['db_name']);
    $parts[] = '--db_user '         . escapeshellarg($opts['db_user']);
    $parts[] = '--db_pass '         . escapeshellarg($opts['db_pass']);

    $parts[] = '--forecast_table '  . escapeshellarg($opts['forecast_table']);
    $parts[] = '--algo_id '         . escapeshellarg((string)$opts['algo_id']);

    // Source query (multi-line ok; escapeshellarg keeps it safe)
    $parts[] = '--source_query '    . escapeshellarg($opts['source_query']);

    $cmd = implode(' ', $parts);

    $descriptorSpec = [
        0 => ['pipe', 'r'],
        1 => ['pipe', 'w'],
        2 => ['pipe', 'w'],
    ];

    $process = proc_open($cmd, $descriptorSpec, $pipes, $projectRoot);

    if (!is_resource($process)) {
        return [
            'exitCode' => -1,
            'stdout'   => '',
            'stderr'   => 'Failed to start Python process.',
            'command'  => $cmd,
        ];
    }

    fclose($pipes[0]);

    $stdout = stream_get_contents($pipes[1]);
    fclose($pipes[1]);

    $stderr = stream_get_contents($pipes[2]);
    fclose($pipes[2]);

    $exitCode = proc_close($process);

    return [
        'exitCode' => $exitCode,
        'stdout'   => $stdout,
        'stderr'   => $stderr,
        'command'  => $cmd,
    ];
}


// -----------------------------------------------------------------------------
// RUN IT
// -----------------------------------------------------------------------------
$opts = [
    'project_root'    => $projectRoot,
    'python_path'     => $pythonPath,
    'runner_path'     => $runnerPath,

    'current_date'    => $currentDate,
    'horizon_weeks'   => $horizonWeeks,

    'db_host'         => $dbHost,
    'db_name'         => $dbName,
    'db_user'         => $dbUser,
    'db_pass'         => $dbPass,

    'forecast_table'  => $forecastTable,
    'algo_id'         => $algoId,

    'source_query'    => $sourceQuery,
];

$result = run_weekly_simulation($opts);


// -----------------------------------------------------------------------------
// SIMPLE HTML RESPONSE
// -----------------------------------------------------------------------------
header('Content-Type: text/html; charset=utf-8');

echo "<h2>Weekly forecast run (as-of {$currentDate}, horizon {$horizonWeeks} weeks)</h2>";

echo "<h3>Command</h3>";
echo "<pre>" . htmlspecialchars($result['command'], ENT_QUOTES, 'UTF-8') . "</pre>";

if ($result['exitCode'] === 0) {
    echo "<h3>Status: ✅ Success (exit code 0)</h3>";
    echo "<h3>Python STDOUT</h3>";
    echo "<pre>" . htmlspecialchars($result['stdout'], ENT_QUOTES, 'UTF-8') . "</pre>";
    if (trim($result['stderr']) !== '') {
        echo "<h3>Python STDERR (warnings/logs)</h3>";
        echo "<pre>" . htmlspecialchars($result['stderr'], ENT_QUOTES, 'UTF-8') . "</pre>";
    }
} else {
    echo "<h3>Status: ❌ FAILED (exit code {$result['exitCode']})</h3>";
    echo "<h3>Python STDERR</h3>";
    echo "<pre>" . htmlspecialchars($result['stderr'], ENT_QUOTES, 'UTF-8') . "</pre>";
    echo "<h3>Python STDOUT</h3>";
    echo "<pre>" . htmlspecialchars($result['stdout'], ENT_QUOTES, 'UTF-8') . "</pre>";
}
