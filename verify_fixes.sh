#!/bin/bash
# Verification script for regression fixes

echo "=========================================="
echo "REGRESSION FIX VERIFICATION"
echo "=========================================="
echo ""

# Test 1: Redzone receiver attribution
echo "Test 1: Redzone receiver text fallback..."
python -m pytest tests/test_redzone_pbp_correctness.py::test_receiver_text_fallback_when_pid_empty -xvs
if [ $? -eq 0 ]; then
    echo "✅ Receiver text fallback test PASSED"
else
    echo "❌ Receiver text fallback test FAILED"
    exit 1
fi
echo ""

# Test 2: Existing Redzone tests
echo "Test 2: All Redzone correctness tests..."
python -m pytest tests/test_redzone_pbp_correctness.py -x
if [ $? -eq 0 ]; then
    echo "✅ All Redzone correctness tests PASSED"
else
    echo "❌ Some Redzone tests FAILED"
    exit 1
fi
echo ""

# Test 3: Check for inert cleanup code
echo "Test 3: Verify inert cleanup code exists..."
if grep -q "cleanupStuckInert" static/paywall.js; then
    echo "✅ Defensive inert cleanup found"
else
    echo "❌ Defensive inert cleanup missing"
    exit 1
fi
echo ""

# Test 4: Check for proper modal replacement cleanup
echo "Test 4: Verify modal replacement cleanup..."
if grep -q "CRITICAL: Properly close existing paywalls" static/paywall.js; then
    echo "✅ Modal replacement cleanup found"
else
    echo "❌ Modal replacement cleanup missing"
    exit 1
fi
echo ""

# Test 5: Check for receiver tracking variables
echo "Test 5: Verify receiver tracking logic..."
if grep -q "has_any_receiver_contrib" utils/redzone_pbp.py; then
    echo "✅ Receiver tracking variables found"
else
    echo "❌ Receiver tracking variables missing"
    exit 1
fi
echo ""

echo "=========================================="
echo "✅ ALL VERIFICATION CHECKS PASSED"
echo "=========================================="
echo ""
echo "Manual verification still required:"
echo "1. Test site-wide interactions (player clicks, search, keyboard)"
echo "2. Test Redzone live feed with NE/SEA game"
echo "3. Test paywall open/close/replace cycles"
echo ""
