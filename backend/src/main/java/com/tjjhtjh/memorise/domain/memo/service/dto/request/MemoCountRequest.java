package com.tjjhtjh.memorise.domain.memo.service.dto.request;

import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

@Getter
@NoArgsConstructor
public class MemoCountRequest {

    private List<Long> itemSeqList;
}
