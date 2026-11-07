# SPEC-13 Station control and telemetry

Bayview Pump Station Upgrade | Contract specification

## 1.1 Control ownership

The station controller owns pump sequencing and local protective interlocks. Telemetry reports status and receives permitted operator requests through the accepted command list. A remote request does not bypass a local protection trip or a maintenance isolation. Keep the accepted program and tag database under revision control.

## 1.2 Input debounce

The PLC alarm-input debounce is 2 seconds. Apply this debounce to the common discrete alarm test path used for the commissioning response check. Do not apply it to motor protective trips. Record the debounce value in the alarm configuration export supplied to the owner.

## 2.1 Local alarm annunciation

The common alarm test path starts at the discrete alarm input, passes through the configured debounce, and then applies the horn delay. Verify both intervals separately during the functional test. Alarm acknowledgement silences the horn but does not clear an active alarm condition. Keep the first-occurrence timestamp in the event log.

## 2.2 Telemetry alarm delay

The telemetry alarm transmission delay is 45 seconds. The delay starts after the station controller declares the alarm active. Use the delay to avoid repeated transient notifications while preserving the local alarm indication. Record a failed transmission as a communications fault at the station.

## 2.3 Remote restart after communications loss

The remote-restart policy after a communications-loss recovery is not selected. The engineer of record shall choose whether operator acknowledgement is required before remote restart is enabled. Local automatic level control remains available during the communications fault. Do not infer the remote-restart policy from the normal pump alternation sequence.

## 3.1 Program acceptance

Demonstrate normal alternation, duty-pump failure transfer, local isolation, and alarm acknowledgement with the owner present. Use simulated inputs where a live wet-well condition would be unsafe. Record the controller revision tested and retain the test input script. Recheck affected functions after any program change.

## 3.2 Telemetry handover

Verify each remote status against the local interface and the field condition. Use the owner test destination for trial notifications. Remove temporary commissioning accounts before handover. Supply the final communications configuration and recovery procedure through the owner approved secure transfer method.
