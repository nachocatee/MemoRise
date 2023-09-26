package com.tjjhtjh.memorise.domain.tag.repository.entity;

import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.team.repository.entity.Team;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.DynamicInsert;
import org.hibernate.annotations.DynamicUpdate;
import org.springframework.data.relational.core.mapping.Table;

@DynamicUpdate
@DynamicInsert
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
@Entity
@Table(name = "tagged_team")
public class TaggedTeam {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "tagged_team_seq")
    private Long taggedTeamSeq;
    @ManyToOne(targetEntity = Team.class, fetch = FetchType.LAZY)
    @JoinColumn(name = "team_seq", referencedColumnName = "team_seq")
    private Team team;
    @ManyToOne(targetEntity = Memo.class, fetch = FetchType.LAZY)
    @JoinColumn(name = "memo_seq", referencedColumnName = "memo_seq")
    private Memo memo;

}
