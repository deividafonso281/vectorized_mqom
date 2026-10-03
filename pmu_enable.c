#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/smp.h>
#include <asm/sysreg.h>
#include <asm/barrier.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("ChatGPT");
MODULE_DESCRIPTION("Enable PMCCNTR_EL0 access from userspace");
MODULE_VERSION("1.0");

static void enable_pmu_on_cpu(void *info)
{
    // PMUSERENR_EL0: Enable EL0 access to PMU
    asm volatile(
        "msr pmuserenr_el0, %0\n"
        :: "r"(1)
    );

    // Enable cycle counter: PMCNTENSET_EL0 bit 31 = 1
    asm volatile(
        "msr pmcntenset_el0, %0\n"
        :: "r"(1 << 31)
    );

    // Enable PMU: PMCR_EL0 |= 1 (Enable)
    asm volatile(
        "mrs x0, pmcr_el0\n"
        "orr x0, x0, #1\n"
        "msr pmcr_el0, x0\n"
        ::: "x0"
    );

    // Ensure everything is applied
    asm volatile("isb");
}

static int __init pmu_enable_init(void)
{
    pr_info("PMU: enabling user access on all CPUs\n");

    // Run the enabling function on all cores
    on_each_cpu(enable_pmu_on_cpu, NULL, 1);

    pr_info("PMU: done\n");
    return 0;
}

static void __exit pmu_enable_exit(void)
{
    pr_info("PMU: module unloaded (PMU stays enabled until reboot)\n");
}

module_init(pmu_enable_init);
module_exit(pmu_enable_exit);

