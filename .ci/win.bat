@echo on
:: this is a nasty hack, and should not be needed, but the env isn't _quite_ right
:: call deactivate
:: call C:\Miniconda\envs\ipyml-base\Scripts\activate
call doit -n4 %%*
call doit %%* || goto :error

goto :EOF

:error
echo Failed with error #%errorlevel%.
exit 1

