package com.tjjhtjh.memorise.domain.team.service.dto.request;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class KickMemberRequest {
    private Long userSeq;
    private Long targetSeq;
}
