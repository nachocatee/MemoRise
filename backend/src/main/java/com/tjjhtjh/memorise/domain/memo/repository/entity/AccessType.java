package com.tjjhtjh.memorise.domain.memo.repository.entity;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor
public enum AccessType {

    OPEN("ACCESSTYPE_OPEN"), RESTRICT("ACCESSTYPE_RESTRICT"), CLOSED("ACCESSTYPE_CLOSED");

    private final String access;
}
